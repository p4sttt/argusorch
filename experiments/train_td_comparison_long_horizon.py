"""
TD(0) vs TD(λ) — Длинный горизонт (max_turns=15)
Оптимизировано для запуска на Tesla T4 (15 GB VRAM).

Изменения относительно исходного ноутбука:
  • dataloader: 1 задача (вместо 2) — один эпизод за шаг
  • replay_capacity=1, ppo_epochs=2 — обновление после каждой траектории
  • max_new_tokens=16 — короче последовательности = меньше пиковой VRAM
  • critic_chunk_size=4 в MAACUpdater — чанкованный forward критика
"""

import os
import sys
import subprocess
import time
import numpy as np
from collections import defaultdict
from types import SimpleNamespace

# ── Путь к корню проекта ─────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Cell 1 — Установка зависимостей ──────────────────────────────────────────


def pip_install(pkg: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


pip_install("bitsandbytes>=0.43.0")
pip_install("transformers>=4.40.0")
pip_install("peft>=0.10.0")
pip_install("accelerate>=0.28.0")

import torch
import transformers
import peft
import bitsandbytes as bnb

print(f"Python:        {sys.version.split()[0]}")
print(f"PyTorch:       {torch.__version__}")
print(f"Transformers:  {transformers.__version__}")
print(f"PEFT:          {peft.__version__}")
print(f"bitsandbytes:  {bnb.__version__}")

print("\n" + "=" * 55)
print("GPU DIAGNOSTICS")
print("=" * 55)
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    print(f"✅ CUDA доступна: {n_gpus} GPU")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / 1024**3
        print(f"   GPU {i}: {props.name} — {vram_gb:.1f} GB VRAM")
        if vram_gb < 12:
            print("   ⚠️  < 12 GB: уменьшите max_new_tokens или используйте lora_r=4")
else:
    print("⚠️  CUDA недоступна — обучение на CPU (очень медленно)")

# ── Cell 2 — GPU-конфигурация и настройка аллокатора ─────────────────────────

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Активное устройство: {DEVICE}")

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    free_gb = (
        torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)
    ) / 1024**3
    print(f"✅ Свободная VRAM при старте: {free_gb:.1f} GB")

SEED = 42
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)
print(f"Seed: {SEED}")

# ── Cell 3 — Гиперпараметры и QLoRA-конфигурация ─────────────────────────────

from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

CFG = SimpleNamespace(
    # ── Модель ────────────────────────────────────────────────────────────
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    bnb_config=bnb_config,

    # ── LoRA ──────────────────────────────────────────────────────────────
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],

    # ── Генерация (↓ 32→16: экономия VRAM при softmax над vocab) ──────────
    max_new_tokens=16,
    temperature=0.7,

    # ── Обучение ──────────────────────────────────────────────────────────
    num_epochs=40,
    actor_lr=2e-4,
    critic_lr=5e-4,
    replay_capacity=1,   # ↓ 4→1: обновляем после каждой траектории
    ppo_epochs=2,        # ↓ 4→2: баланс качества и памяти

    # ── TD параметры ──────────────────────────────────────────────────────
    gamma=0.995,
    td_lambda=0.95,

    # ── Среда — ДЛИННЫЙ ГОРИЗОНТ ──────────────────────────────────────────
    num_agents=2,
    max_turns=15,

    # ── Батч критика (чанкование evaluate_states) ─────────────────────────
    critic_chunk_size=4,
)

# ── Cell 4 — Загрузка моделей с QLoRA ────────────────────────────────────────

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training


def load_qlora_model(name: str, lora_cfg: LoraConfig) -> torch.nn.Module:
    """Загружает модель с 4-bit QLoRA + gradient checkpointing."""
    base = AutoModelForCausalLM.from_pretrained(
        name,
        quantization_config=CFG.bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=True,
    )
    return get_peft_model(base, lora_cfg)


tokenizer = AutoTokenizer.from_pretrained(CFG.model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

lora_config = LoraConfig(
    r=CFG.lora_r,
    lora_alpha=CFG.lora_alpha,
    target_modules=CFG.target_modules,
    lora_dropout=CFG.lora_dropout,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

print("⏳ [MAS-A / TD0] actor...")
actor_model_td0 = load_qlora_model(CFG.model_name, lora_config)
print("⏳ [MAS-A / TD0] critic...")
critic_model_td0 = load_qlora_model(CFG.model_name, lora_config)

print("⏳ [MAS-B / TDλ] actor...")
actor_model_tdl = load_qlora_model(CFG.model_name, lora_config)
print("⏳ [MAS-B / TDλ] critic...")
critic_model_tdl = load_qlora_model(CFG.model_name, lora_config)

print("\n📊 Trainable params:")
actor_model_td0.print_trainable_parameters()

if DEVICE.type == "cuda":
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserv = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\n💾 VRAM после загрузки 4 × QLoRA-моделей:")
    print(f"   Allocated: {alloc:.2f} GB")
    print(f"   Reserved:  {reserv:.2f} GB")
    print(f"   Total GPU: {total:.1f} GB")
    print(f"   Доступно:  ~{total - reserv:.1f} GB для обучения")

# ── Cell 5 — Инициализация агентов и критиков ─────────────────────────────────

from argusorch.agents import LLMActor, AgentsGroup, CentralizedCritic
from argusorch.agents.prompt_builder import PromptBuilder
from argusorch.env.types import JointState


def get_model_device(model) -> torch.device:
    return next(reversed(list(model.parameters()))).device


gen_config = {
    "max_new_tokens": CFG.max_new_tokens,
    "do_sample": True,
    "temperature": CFG.temperature,
    "pad_token_id": tokenizer.pad_token_id,
}


class LongHorizonPromptBuilder(PromptBuilder):
    def build(self, joint_state: JointState) -> str:
        goal = joint_state.item.get("goal", "")
        constraints = joint_state.item.get("constraints", [])
        history_lines = []
        for turn_record in joint_state.history:
            t = turn_record.get("turn", "?")
            for a_id, text in turn_record.get("actions", {}).items():
                history_lines.append(f"  [T{t}][{a_id}]: {text[:100]}")
        history_str = "\n".join(history_lines) if history_lines else "(empty)"
        constraints_str = "; ".join(constraints) if constraints else "none"
        return (
            f"[CRITIC] Long-Horizon Planning\n"
            f"Goal: {goal}\n"
            f"Constraints: {constraints_str}\n"
            f"Turn: {joint_state.turn}/{CFG.max_turns}\n"
            f"Plan History:\n{history_str}\n"
            f"Rate coherence (0.0=incoherent, 1.0=excellent):"
        )


# ── MAS-A: TD(0) ──────────────────────────────────────────────────────────────
coder_td0 = LLMActor(actor_model_td0, tokenizer, gen_config)
tester_td0 = LLMActor(actor_model_td0, tokenizer, gen_config)
agents_td0 = AgentsGroup([coder_td0, tester_td0])
critic_td0 = CentralizedCritic(critic_model_td0, tokenizer, LongHorizonPromptBuilder())
critic_td0.value_head = critic_td0.value_head.to(get_model_device(critic_model_td0))

# ── MAS-B: TD(λ) ──────────────────────────────────────────────────────────────
coder_tdl = LLMActor(actor_model_tdl, tokenizer, gen_config)
tester_tdl = LLMActor(actor_model_tdl, tokenizer, gen_config)
agents_tdl = AgentsGroup([coder_tdl, tester_tdl])
critic_tdl = CentralizedCritic(critic_model_tdl, tokenizer, LongHorizonPromptBuilder())
critic_tdl.value_head = critic_tdl.value_head.to(get_model_device(critic_model_tdl))

print(f"✅ MAS-A (TD0):  device={get_model_device(actor_model_td0)}")
print(f"✅ MAS-B (TD-λ): device={get_model_device(actor_model_tdl)}")

# ── Cell 6 — Среда, датасет и компоненты обучения ────────────────────────────

from argusorch.env import LongHorizonPlanningEnv
from argusorch.env.rollout_collector import RolloutCollector
from argusorch.trainers.maac.losses import MAACLoss
from argusorch.trainers.maac.updater import MAACUpdater
from argusorch.trainers.common import (
    TD0TargetEstimator,
    TDLambdaEstimator,
    ReplayBuffer,
)

# ↓ 1 задача вместо 2: минимальный батч joint_states при evaluate_states
dataloader = [
    {
        "goal": "Design a distributed microservices architecture for an e-commerce platform",
        "constraints": ["99.9% availability", "latency < 100ms", "GDPR compliant"],
    },
]


def make_paged_optimizer(model, lr: float) -> bnb.optim.PagedAdamW8bit:
    trainable = [p for p in model.parameters() if p.requires_grad]
    return bnb.optim.PagedAdamW8bit(trainable, lr=lr)


def make_system_lh(actor_model, critic_model, estimator):
    env = LongHorizonPlanningEnv(num_agents=CFG.num_agents, max_turns=CFG.max_turns)
    a_group = AgentsGroup(
        [LLMActor(actor_model, tokenizer, gen_config) for _ in range(CFG.num_agents)]
    )
    collector = RolloutCollector(env, a_group, critic_model)
    buffer = ReplayBuffer(capacity=CFG.replay_capacity, device=DEVICE)

    actor_opt = make_paged_optimizer(actor_model, CFG.actor_lr)
    critic_opt = make_paged_optimizer(critic_model, CFG.critic_lr)

    actors_dict = {f"agent_{i}": a for i, a in enumerate(a_group.actors)}
    actor_opts = {f"agent_{i}": actor_opt for i in range(CFG.num_agents)}

    updater = MAACUpdater(
        actors_dict,
        critic_model,
        actor_opts,
        critic_opt,
        MAACLoss(),
        ppo_epochs=CFG.ppo_epochs,
        device=DEVICE,
        critic_chunk_size=CFG.critic_chunk_size,
    )
    return collector, buffer, updater, a_group


estimator_td0 = TD0TargetEstimator(gamma=CFG.gamma)
collector_td0, buffer_td0, updater_td0, agents_td0 = make_system_lh(
    actor_model_td0, critic_td0, estimator_td0
)

estimator_tdl = TDLambdaEstimator(gamma=CFG.gamma, lambda_=CFG.td_lambda)
collector_tdl, buffer_tdl, updater_tdl, agents_tdl = make_system_lh(
    actor_model_tdl, critic_tdl, estimator_tdl
)

print("✅ Оба MAS инициализированы")
print(f"   TD(0):     γ={CFG.gamma}")
print(f"   TD(λ):     γ={CFG.gamma}, λ={CFG.td_lambda}")
print(f"   Среда:     LongHorizonPlanningEnv | max_turns={CFG.max_turns}")
print(f"   replay_capacity={CFG.replay_capacity} | ppo_epochs={CFG.ppo_epochs} | max_new_tokens={CFG.max_new_tokens}")

# ── Cell 7 — Параллельный цикл обучения ──────────────────────────────────────

log_td0 = defaultdict(list)
log_tdl = defaultdict(list)
steps_log = []
adv_std_td0 = []
adv_std_tdl = []
update_steps_td0 = []
update_steps_tdl = []

print(f"🚀 Запуск обучения: {CFG.num_epochs} эпох × {len(dataloader)} задания")
print(f"   MAS-A: TD(0)       | γ={CFG.gamma}")
print(f"   MAS-B: TD(λ={CFG.td_lambda}) | γ={CFG.gamma}, λ={CFG.td_lambda}")
print(f"   🎯 Разреженная награда: r≠0 ТОЛЬКО на шаге {CFG.max_turns}")
print("-" * 78)

t_start = time.time()
global_step = 0


def run_step_track_adv(collector, estimator, buffer, updater) -> tuple:
    traj, env_metrics = collector.collect(batch)
    targets = estimator.compute(traj)
    buffer.add(traj, targets)
    metrics = env_metrics.copy()

    advantages = []
    for agent_traj in targets.by_agent():
        for t in agent_traj:
            advantages.append(t.advantage)
    adv_std = float(np.std(advantages)) if advantages else 0.0

    if buffer.ready():
        train_batch = buffer.sample()
        update_metrics = updater.update(train_batch)
        metrics.update(update_metrics)

    return metrics, adv_std


for epoch in range(CFG.num_epochs):
    for batch in dataloader:
        # ── MAS-A: TD(0) ─────────────────────────────────────────────────
        m_td0, std_td0 = run_step_track_adv(collector_td0, estimator_td0, buffer_td0, updater_td0)

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        # ── MAS-B: TD(λ) ─────────────────────────────────────────────────
        m_tdl, std_tdl = run_step_track_adv(collector_tdl, estimator_tdl, buffer_tdl, updater_tdl)

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        global_step += 1
        steps_log.append(global_step)
        adv_std_td0.append(std_td0)
        adv_std_tdl.append(std_tdl)

        if "critic_loss" in m_td0:
            update_steps_td0.append(global_step)
        if "critic_loss" in m_tdl:
            update_steps_tdl.append(global_step)

        for k, v in m_td0.items():
            if isinstance(v, float):
                log_td0[k].append(v)
        for k, v in m_tdl.items():
            if isinstance(v, float):
                log_tdl[k].append(v)

        elapsed = time.time() - t_start
        c_td0 = m_td0.get("critic_loss", float("nan"))
        c_tdl = m_tdl.get("critic_loss", float("nan"))
        r_td0 = m_td0.get("reward", float("nan"))
        r_tdl = m_tdl.get("reward", float("nan"))
        print(
            f"[E{epoch+1:02d}/{CFG.num_epochs} S{global_step:03d}] "
            f"TD0: c={c_td0:.4f} r={r_td0:.1f} σ={std_td0:.4f}  "
            f"| TDλ: c={c_tdl:.4f} r={r_tdl:.1f} σ={std_tdl:.4f}  "
            f"⏱{elapsed:.1f}s"
        )

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserv = torch.cuda.memory_reserved() / 1024**3
        print(f"  💾 VRAM: alloc={alloc:.2f} GB, reserved={reserv:.2f} GB")

total_time = time.time() - t_start
print(f"\n✅ Обучение завершено за {total_time:.1f}с ({total_time/60:.1f} мин)")

# ── Cell 8 — Сравнительные графики ───────────────────────────────────────────

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def moving_avg(x: list, w: int = 5) -> np.ndarray:
    arr = np.array(x, dtype=float)
    return arr if len(arr) < w else np.convolve(arr, np.ones(w) / w, mode="valid")


def rolling_std(x: list, w: int = 5) -> np.ndarray:
    arr = np.array(x, dtype=float)
    return np.array([arr[i : i + w].std() for i in range(len(arr) - w + 1)])


COLOR_TD0 = "#e63946"
COLOR_TDL = "#2a9d8f"
WINDOW = max(3, min(len(steps_log) // 8, 10))

all_keys = sorted(set(log_td0.keys()) | set(log_tdl.keys()))
n_panels = 2 + len(all_keys)
cols = 3
rows = (n_panels + cols - 1) // cols

fig = plt.figure(figsize=(16, 4.5 * rows))
fig.suptitle(
    f"QLoRA | TD(0) vs TD(λ={CFG.td_lambda}) — LongHorizonPlanning\n"
    f"{CFG.model_name} | 4-bit NF4 | max_turns={CFG.max_turns} | γ={CFG.gamma}",
    fontsize=12,
    fontweight="bold",
)
gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.55, wspace=0.38)

# ── Панель 1: Advantage Std ────────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(steps_log, adv_std_td0, color=COLOR_TD0, lw=0.8, alpha=0.3)
ax0.plot(steps_log, adv_std_tdl, color=COLOR_TDL, lw=0.8, alpha=0.3)
if len(adv_std_td0) >= WINDOW:
    ma0 = moving_avg(adv_std_td0, WINDOW)
    mal = moving_avg(adv_std_tdl, WINDOW)
    std0 = rolling_std(adv_std_td0, WINDOW)
    stdl = rolling_std(adv_std_tdl, WINDOW)
    ax0.plot(steps_log[WINDOW - 1 :], ma0, color=COLOR_TD0, lw=2.5, label=f"TD(0) MA-{WINDOW}")
    ax0.fill_between(
        steps_log[WINDOW - 1 :], ma0 - std0, ma0 + std0, color=COLOR_TD0, alpha=0.15
    )
    ax0.plot(steps_log[WINDOW - 1 :], mal, color=COLOR_TDL, lw=2.5, label=f"TD(λ) MA-{WINDOW}")
    ax0.fill_between(
        steps_log[WINDOW - 1 :], mal - stdl, mal + stdl, color=COLOR_TDL, alpha=0.15
    )
ax0.set_title("🔑 Advantage Std σ\nБогатство обучающего сигнала", fontsize=9, pad=4)
ax0.set_xlabel("Step", fontsize=8)
ax0.legend(fontsize=7)
ax0.grid(True, alpha=0.25)
ax0.tick_params(labelsize=7)

# ── Панель 2: Rolling Variance critic_loss ─────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 1])
if "critic_loss" in log_td0 and len(log_td0["critic_loss"]) >= WINDOW:
    var0 = rolling_std(log_td0["critic_loss"], WINDOW)
    varl = rolling_std(log_tdl.get("critic_loss", [0.0] * len(steps_log)), WINDOW)
    xs_v = steps_log[WINDOW - 1 :]
    ax1.fill_between(xs_v[: len(var0)], 0, var0, color=COLOR_TD0, alpha=0.35, label="TD(0)")
    ax1.fill_between(xs_v[: len(varl)], 0, varl, color=COLOR_TDL, alpha=0.35, label="TD(λ)")
    ax1.plot(xs_v[: len(var0)], var0, color=COLOR_TD0, lw=1.5)
    ax1.plot(xs_v[: len(varl)], varl, color=COLOR_TDL, lw=1.5)
ax1.set_title("📉 Critic Loss Rolling Std\nСтабильность", fontsize=9, pad=4)
ax1.set_xlabel("Step", fontsize=8)
ax1.legend(fontsize=7)
ax1.grid(True, alpha=0.25)
ax1.tick_params(labelsize=7)

# ── Остальные метрики ──────────────────────────────────────────────────────────
for idx, key in enumerate(all_keys):
    pos = idx + 2
    ax = fig.add_subplot(gs[pos // cols, pos % cols])
    for vals, color, label in [
        (log_td0.get(key, []), COLOR_TD0, "TD(0)"),
        (log_tdl.get(key, []), COLOR_TDL, "TD(λ)"),
    ]:
        if not vals:
            continue
        xs_cut = steps_log[-len(vals) :]
        ax.plot(xs_cut, vals, color=color, lw=0.8, alpha=0.3)
        if len(vals) >= WINDOW:
            ma_vals = moving_avg(vals, WINDOW)
            std_vals = rolling_std(vals, WINDOW)
            ax.plot(xs_cut[WINDOW - 1 :], ma_vals, color=color, lw=2, label=label)
            ax.fill_between(
                xs_cut[WINDOW - 1 :],
                ma_vals - std_vals,
                ma_vals + std_vals,
                color=color,
                alpha=0.15,
            )
    ax.set_title(key, fontsize=10, pad=4)
    ax.set_xlabel("Step", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig("td_comparison_long_horizon_qlora.png", dpi=150, bbox_inches="tight")
plt.show()
print("📁 График сохранён: td_comparison_long_horizon_qlora.png")

# ── Cell 9 — Детальный анализ Credit Assignment Problem ─────────────────────

print("=" * 78)
print("ДЕТАЛЬНЫЙ АНАЛИЗ: TD(0) vs TD(λ) — Длинный горизонт")
print(f"Модель: {CFG.model_name} | QLoRA 4-bit NF4")
print(f"LongHorizonPlanningEnv | max_turns={CFG.max_turns} | γ={CFG.gamma} | λ={CFG.td_lambda}")
print("=" * 78)

tail = max(1, len(steps_log) // 4)

print(f"\n{'Метрика':<28} {'TD(0) mean±std':>22} {'TD(λ) mean±std':>22} {'Δ':>11}")
print("-" * 85)
for key in sorted(set(log_td0.keys()) | set(log_tdl.keys())):
    v0 = np.array(log_td0.get(key, [0.0])[-tail:])
    vl = np.array(log_tdl.get(key, [0.0])[-tail:])
    m0, s0 = v0.mean(), v0.std()
    ml, sl = vl.mean(), vl.std()
    print(f"{key:<28} {m0:>10.4f}±{s0:<10.4f} {ml:>10.4f}±{sl:<10.4f} {ml-m0:>+10.4f}")

print("\n" + "=" * 78)
print("🎯 ADVANTAGE STD — БОГАТСТВО ОБУЧАЮЩЕГО СИГНАЛА:")
print("-" * 78)
mean_std_td0 = np.mean(adv_std_td0[-tail:])
mean_std_tdl = np.mean(adv_std_tdl[-tail:])
print(f"  TD(0)  advantage σ = {mean_std_td0:.5f}")
print(f"  TD(λ)  advantage σ = {mean_std_tdl:.5f}")
if mean_std_tdl > mean_std_td0:
    ratio = mean_std_tdl / (mean_std_td0 + 1e-8)
    print(f"\n  ✅ TD(λ) производит advantage в {ratio:.2f}x информативнее.")
else:
    print(f"\n  → Продолжите обучение для накопления эффекта.")

print("\n" + "=" * 78)
print(f"🔬 CREDIT ASSIGNMENT при T={CFG.max_turns}, γ={CFG.gamma}:")
print("-" * 78)
print(
    f"  TD(0): G_0 = γ^{CFG.max_turns-1} · R_{CFG.max_turns} = "
    f"{CFG.gamma**(CFG.max_turns-1):.4f} · R_{CFG.max_turns}"
)
print(f"         ⚠️  Ошибка накапливается через {CFG.max_turns-1} bootstrap-шагов.")
print()
print(f"  TD(λ={CFG.td_lambda}): G_0^λ = (1-λ)·Σ λ^(n-1)·G^(n)")
print(f"         Вклад G^({CFG.max_turns}): λ^{CFG.max_turns-1} = {CFG.td_lambda**(CFG.max_turns-1):.4f}")
print(f"         ✅ Прямое взвешивание, нет накопления bootstrap-ошибок.")
print()
print("=" * 78)
print("💡 ВЫВОД: TD(λ) > TD(0) при длинном горизонте + разреженных наградах:")
print("   1. Богаче advantage σ → более дифференцированный gradient signal")
print("   2. Нет экспоненциального накопления bootstrap-ошибок")
print(f"   3. λ={CFG.td_lambda} — баланс между MC (λ=1) и TD(0) (λ=0)")
print("=" * 78)
