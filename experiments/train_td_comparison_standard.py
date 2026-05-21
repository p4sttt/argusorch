"""
TD(0) vs TD(λ) — Стандартный горизонт (max_turns=5)
Оптимизировано для запуска на Tesla T4 (15 GB VRAM).

Изменения относительно исходного ноутбука:
  • dataloader: 2 задачи (вместо 9) — меньший батч критика на каждом шаге
  • replay_capacity=1, ppo_epochs=1 — обновление после каждой траектории
  • max_new_tokens=16 — короче последовательности = меньше пиковой VRAM
  • critic_chunk_size=4 в MAACUpdater — чанкованный forward критика
  • Отображение дисперсии advantage (adv_std) — как в long_horizon скрипте
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

# ── Cell 2 — GPU-конфигурация ─────────────────────────────────────────────────

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
print("✅ PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Активное устройство: {DEVICE}")

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    free_gb = (
        torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)
    ) / 1024**3
    print(f"✅ Свободная VRAM: {free_gb:.1f} GB")

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
    target_modules="all-linear",

    # ── Генерация (↓ 32→16 токенов: экономия VRAM при softmax над vocab) ──
    max_new_tokens=16,
    temperature=0.7,

    # ── Обучение ──────────────────────────────────────────────────────────
    num_epochs=15,
    actor_lr=2e-4,
    critic_lr=5e-4,
    replay_capacity=1,   # ↓ 2→1: обновляем после каждой траектории
    ppo_epochs=1,        # ↓ 2→1: один проход PPO на T4

    # ── TD параметры ──────────────────────────────────────────────────────
    gamma=0.99,
    td_lambda=0.95,

    # ── Среда — СТАНДАРТНЫЙ ГОРИЗОНТ ──────────────────────────────────────
    num_agents=2,
    max_turns=5,

    # ── Батч критика (чанкование evaluate_states) ─────────────────────────
    critic_chunk_size=4,
)

# ── Cell 4 — Загрузка моделей с QLoRA ────────────────────────────────────────

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

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


def load_qlora_model(name: str, lora_cfg: LoraConfig) -> torch.nn.Module:
    base = AutoModelForCausalLM.from_pretrained(
        name,
        quantization_config=CFG.bnb_config,
        device_map={"": "cuda:0"} if torch.cuda.is_available() else "auto",
        trust_remote_code=True,
    )
    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=True,
    )
    return get_peft_model(base, lora_cfg)


print("⏳ [MAS-A / TD0] Загрузка actor...")
actor_model_td0 = load_qlora_model(CFG.model_name, lora_config)
print("⏳ [MAS-A / TD0] Загрузка critic...")
critic_model_td0 = load_qlora_model(CFG.model_name, lora_config)

print("⏳ [MAS-B / TDλ] Загрузка actor...")
actor_model_tdl = load_qlora_model(CFG.model_name, lora_config)
print("⏳ [MAS-B / TDλ] Загрузка critic...")
critic_model_tdl = load_qlora_model(CFG.model_name, lora_config)

print("\n📊 Trainable params (одинаково для всех 4 моделей):")
actor_model_td0.print_trainable_parameters()

if DEVICE.type == "cuda":
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserv = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\n💾 VRAM после загрузки 4 моделей:")
    print(f"   Allocated: {alloc:.2f} GB")
    print(f"   Reserved:  {reserv:.2f} GB")
    print(f"   Total:     {total:.1f} GB")

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


class CodeCollabPromptBuilder(PromptBuilder):
    def build(self, joint_state: JointState) -> str:
        task = joint_state.item.get("task", "")
        history_lines = []
        for turn in joint_state.history:
            for a_id, text in turn.get("actions", {}).items():
                history_lines.append(f"  [{a_id}]: {text[:200]}")
        history_str = "\n".join(history_lines) if history_lines else "(empty)"
        return (
            f"[CRITIC] Task: {task}\n"
            f"Turn: {joint_state.turn}\n"
            f"History:\n{history_str}\n"
            f"Rate collaboration quality (0.0=bad, 1.0=excellent):"
        )


# ── MAS-A: TD(0) ──────────────────────────────────────────────────────────────
coder_td0 = LLMActor(actor_model_td0, tokenizer, gen_config)
tester_td0 = LLMActor(actor_model_td0, tokenizer, gen_config)
agents_td0 = AgentsGroup([coder_td0, tester_td0])
critic_td0 = CentralizedCritic(critic_model_td0, tokenizer, CodeCollabPromptBuilder())
critic_td0.value_head = critic_td0.value_head.to(get_model_device(critic_model_td0))

# ── MAS-B: TD(λ) ──────────────────────────────────────────────────────────────
coder_tdl = LLMActor(actor_model_tdl, tokenizer, gen_config)
tester_tdl = LLMActor(actor_model_tdl, tokenizer, gen_config)
agents_tdl = AgentsGroup([coder_tdl, tester_tdl])
critic_tdl = CentralizedCritic(critic_model_tdl, tokenizer, CodeCollabPromptBuilder())
critic_tdl.value_head = critic_tdl.value_head.to(get_model_device(critic_model_tdl))

print(f"✅ MAS-A (TD0):  {len(agents_td0.actors)} агента (Parameter Sharing)")
print(f"✅ MAS-B (TD-λ): {len(agents_tdl.actors)} агента (Parameter Sharing)")

# ── Cell 6 — Среда, датасет и компоненты обучения ────────────────────────────

from argusorch.env import CodeCollabEnv
from argusorch.env.rollout_collector import RolloutCollector
from argusorch.trainers.maac.losses import MAACLoss
from argusorch.trainers.maac.updater import MAACUpdater
from argusorch.trainers.common import (
    TD0TargetEstimator,
    TDLambdaEstimator,
    ReplayBuffer,
)

# ↓ 2 задачи вместо 9: меньший батч joint_states при evaluate_states → меньше VRAM
dataloader = [
    {"task": "Write a Python function to sort a list using bubble sort", "unit_tests": []},
    {"task": "Write a Python function to compute the Fibonacci sequence", "unit_tests": []},
]


def make_paged_optimizer(model, lr: float) -> bnb.optim.PagedAdamW8bit:
    trainable = [p for p in model.parameters() if p.requires_grad]
    return bnb.optim.PagedAdamW8bit(trainable, lr=lr)


def make_system(actor_model, critic, estimator):
    env = CodeCollabEnv(num_agents=CFG.num_agents, max_turns=CFG.max_turns)
    a_group = AgentsGroup(
        [LLMActor(actor_model, tokenizer, gen_config) for _ in range(CFG.num_agents)]
    )
    collector = RolloutCollector(env, a_group, critic)
    buffer = ReplayBuffer(capacity=CFG.replay_capacity, device=DEVICE)
    actor_opt = make_paged_optimizer(actor_model, CFG.actor_lr)
    critic_opt = make_paged_optimizer(critic, CFG.critic_lr)
    actors_dict = {f"agent_{i}": a for i, a in enumerate(a_group.actors)}
    actor_opts = {f"agent_{i}": actor_opt for i in range(CFG.num_agents)}

    updater = MAACUpdater(
        actors_dict,
        critic,
        actor_opts,
        critic_opt,
        MAACLoss(),
        ppo_epochs=CFG.ppo_epochs,
        device=DEVICE,
        critic_chunk_size=CFG.critic_chunk_size,
    )
    return collector, buffer, updater, a_group


estimator_td0 = TD0TargetEstimator(gamma=CFG.gamma)
collector_td0, buffer_td0, updater_td0, agents_td0 = make_system(
    actor_model_td0, critic_td0, estimator_td0
)

estimator_tdl = TDLambdaEstimator(gamma=CFG.gamma, lambda_=CFG.td_lambda)
collector_tdl, buffer_tdl, updater_tdl, agents_tdl = make_system(
    actor_model_tdl, critic_tdl, estimator_tdl
)

# ── Cell 7 — Параллельный цикл обучения обоих MAS ────────────────────────────

log_td0 = defaultdict(list)
log_tdl = defaultdict(list)
steps_log = []
adv_std_td0 = []   # дисперсия advantage — богатство обучающего сигнала
adv_std_tdl = []
update_steps_td0 = []
update_steps_tdl = []

print(f"🚀 Запуск обучения: {CFG.num_epochs} эпох × {len(dataloader)} задания")
print(f"   MAS-A: TD(0)       | MAS-B: TD(λ={CFG.td_lambda})")
print(f"   replay_capacity={CFG.replay_capacity} | ppo_epochs={CFG.ppo_epochs} | max_new_tokens={CFG.max_new_tokens}")
print("-" * 70)

t_start = time.time()
global_step = 0


def run_step(collector, estimator, buffer, updater) -> tuple:
    traj, env_metrics = collector.collect(batch)
    targets = estimator.compute(traj)
    buffer.add(traj, targets)
    metrics = env_metrics.copy()

    # Собираем advantage для отслеживания дисперсии обучающего сигнала
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
        m_td0, std_td0 = run_step(collector_td0, estimator_td0, buffer_td0, updater_td0)

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        m_tdl, std_tdl = run_step(collector_tdl, estimator_tdl, buffer_tdl, updater_tdl)

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
            f"TD0: crit={c_td0:.4f} r={r_td0:.3f} σ={std_td0:.4f}  "
            f"| TDλ: crit={c_tdl:.4f} r={r_tdl:.3f} σ={std_tdl:.4f}  "
            f"⏱{elapsed:.1f}s"
        )

    # Очистка памяти только раз в эпоху
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserv = torch.cuda.memory_reserved() / 1024**3
        print(f"  💾 VRAM: allocated={alloc:.2f} GB, reserved={reserv:.2f} GB")

total_time = time.time() - t_start
print(f"\n✅ Обучение завершено за {total_time:.1f}с ({total_time/60:.1f} мин)")

# ── Cell 8 — Сравнительные графики: TD(0) vs TD(λ) ───────────────────────────

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
# +2 для панелей Advantage Std и Critic Loss Rolling Std
n_panels = 2 + len(all_keys)
cols = 3
rows = (n_panels + cols - 1) // cols

fig = plt.figure(figsize=(16, 4.5 * rows))
fig.suptitle(
    f"QLoRA | TD(0) vs TD(λ={CFG.td_lambda}) — CodeCollab (max_turns={CFG.max_turns})\n"
    f"Model: {CFG.model_name} | 4-bit NF4",
    fontsize=12,
    fontweight="bold",
)
gs = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.55, wspace=0.38)

# ── Панель 1: Advantage Std (дисперсия обучающего сигнала) ────────────────────
ax0 = fig.add_subplot(gs[0, 0])
ax0.plot(steps_log, adv_std_td0, color=COLOR_TD0, lw=0.8, alpha=0.3)
ax0.plot(steps_log, adv_std_tdl, color=COLOR_TDL, lw=0.8, alpha=0.3)
if len(adv_std_td0) >= WINDOW:
    ma0 = moving_avg(adv_std_td0, WINDOW)
    mal = moving_avg(adv_std_tdl, WINDOW)
    std0 = rolling_std(adv_std_td0, WINDOW)
    stdl = rolling_std(adv_std_tdl, WINDOW)
    ax0.plot(steps_log[WINDOW - 1 :], ma0, color=COLOR_TD0, lw=2.5, label=f"TD(0) MA-{WINDOW}")
    ax0.fill_between(steps_log[WINDOW - 1 :], ma0 - std0, ma0 + std0, color=COLOR_TD0, alpha=0.15)
    ax0.plot(steps_log[WINDOW - 1 :], mal, color=COLOR_TDL, lw=2.5, label=f"TD(λ) MA-{WINDOW}")
    ax0.fill_between(steps_log[WINDOW - 1 :], mal - stdl, mal + stdl, color=COLOR_TDL, alpha=0.15)
ax0.set_title("🔑 Advantage Std σ\nДисперсия обучающего сигнала", fontsize=9, pad=4)
ax0.set_xlabel("Step", fontsize=8)
ax0.legend(fontsize=7)
ax0.grid(True, alpha=0.25)
ax0.tick_params(labelsize=7)

# ── Панель 2: Rolling Std critic_loss ─────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 1])
if "critic_loss" in log_td0 and len(log_td0["critic_loss"]) >= WINDOW:
    var0 = rolling_std(log_td0["critic_loss"], WINDOW)
    varl = rolling_std(log_tdl.get("critic_loss", [0.0] * len(steps_log)), WINDOW)
    xs_v = steps_log[WINDOW - 1 :]
    ax1.fill_between(xs_v[: len(var0)], 0, var0, color=COLOR_TD0, alpha=0.35, label="TD(0)")
    ax1.fill_between(xs_v[: len(varl)], 0, varl, color=COLOR_TDL, alpha=0.35, label="TD(λ)")
    ax1.plot(xs_v[: len(var0)], var0, color=COLOR_TD0, lw=1.5)
    ax1.plot(xs_v[: len(varl)], varl, color=COLOR_TDL, lw=1.5)
ax1.set_title("📉 Critic Loss Rolling Std\nСтабильность обучения", fontsize=9, pad=4)
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
plt.savefig("td_comparison_standard_qlora.png", dpi=150, bbox_inches="tight")
plt.show()
print("📁 График сохранён: td_comparison_standard_qlora.png")

# ── Cell 9 — Итоговая таблица и анализ Bias-Variance Tradeoff ────────────────

print("=" * 70)
print("ИТОГОВОЕ СРАВНЕНИЕ: TD(0) vs TD(λ) — Стандартный горизонт")
print(f"Модель: {CFG.model_name} | QLoRA 4-bit NF4")
print(f"Среда: CodeCollabEnv | max_turns={CFG.max_turns} | epochs={CFG.num_epochs}")
print("=" * 70)

tail = max(1, len(steps_log) // 4)

print(f"\n{'Метрика':<25} {'TD(0) mean±std':>22} {'TD(λ) mean±std':>22} {'Δ':>12}")
print("-" * 83)
for key in sorted(set(log_td0.keys()) | set(log_tdl.keys())):
    v0 = np.array(log_td0.get(key, [0.0])[-tail:])
    vl = np.array(log_tdl.get(key, [0.0])[-tail:])
    m0, s0 = v0.mean(), v0.std()
    ml, sl = vl.mean(), vl.std()
    print(f"{key:<25} {m0:>10.4f}±{s0:<10.4f} {ml:>10.4f}±{sl:<10.4f} {ml-m0:>+11.4f}")

print("\n" + "=" * 70)
print("📐 ДИСПЕРСИЯ ADVANTAGE (богатство обучающего сигнала):")
print("-" * 70)
mean_std_td0 = np.mean(adv_std_td0[-tail:])
mean_std_tdl = np.mean(adv_std_tdl[-tail:])
print(f"  TD(0)  advantage σ = {mean_std_td0:.5f}")
print(f"  TD(λ)  advantage σ = {mean_std_tdl:.5f}")
if mean_std_tdl > mean_std_td0:
    ratio = mean_std_tdl / (mean_std_td0 + 1e-8)
    print(f"\n  ✅ TD(λ) производит advantage в {ratio:.2f}x информативнее.")
else:
    print(f"\n  → При коротком горизонте TD(0) и TD(λ) близки.")
    print(f"     Преимущество TD(λ) раскрывается при длинном горизонте (см. long_horizon).")

print("\n" + "=" * 70)
print("📐 BIAS-VARIANCE АНАЛИЗ (critic_loss):")
print("-" * 70)
if "critic_loss" in log_td0 and "critic_loss" in log_tdl:
    std_td0 = np.array(log_td0["critic_loss"][-tail:]).std()
    std_tdl = np.array(log_tdl["critic_loss"][-tail:]).std()
    print(f"  critic_loss σ [TD(0)]: {std_td0:.5f}")
    print(f"  critic_loss σ [TD(λ)]: {std_tdl:.5f}")
    if std_tdl < std_td0:
        print(f"\n  ✅ TD(λ) снижает дисперсию critic_loss на {(std_td0-std_tdl)/std_td0*100:.1f}%")
    else:
        print(f"\n  → При коротком горизонте дисперсия critic_loss схожа.")
        print(f"     → Для яркого эффекта см. long_horizon!")

print("\n" + "=" * 70)
print("🔬 МАТЕМАТИКА:")
print("  TD(0):  G_t = r_t + γ·V(s_{t+1})")
print("          → Bootstrap с одним шагом: смещение от неточной V")
print("  TD(λ):  G_t^λ = (1-λ)·Σ_{n≥1} λ^{n-1}·G_t^(n)")
print(f"          → λ={CFG.td_lambda}: взвешенная сумма всех n-шаговых возвратов")
print("          → Балансирует bias (λ→0) и variance (λ→1)")
print("=" * 70)
