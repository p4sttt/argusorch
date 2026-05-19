import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from types import SimpleNamespace

from argusorch.agents import LLMActor, AgentsGroup, CentralizedCritic
from argusorch.env import CodeCollabEnv, RolloutCollector
from argusorch.trainers.maac import MAACTrainer, MAACUpdater
from argusorch.trainers.maac.losses import MAACLoss
from argusorch.trainers.common import GAEEstimator, ReplayBuffer


from argusorch.trainers.common.loggers import ConsoleLogger

def main():
    # 1. Загрузка моделей и применение PEFT/LoRA адаптеров
    model_name = "Qwen/Qwen1.5-0.5B"  # Маленькая модель для примера
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    actor_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    )
    critic_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    actor_model = get_peft_model(actor_model, lora_config)
    critic_model = get_peft_model(critic_model, lora_config)
    print("Актор обучает параметров:")
    actor_model.print_trainable_parameters()

    # 2. Инициализация Агентов
    gen_config = {"max_new_tokens": 50, "do_sample": True, "temperature": 0.7}
    coder = LLMActor(actor_model, tokenizer, gen_config)
    tester = LLMActor(actor_model, tokenizer, gen_config)  # Шарят веса, но роли разные

    agents_group = AgentsGroup([coder, tester])

    # 3. Инициализация Критика
    class DummyPromptBuilder:
        def build(self, joint_state):
            return str(joint_state)

    critic = CentralizedCritic(critic_model, tokenizer, DummyPromptBuilder())

    # 4. Инициализация среды, Оптимизаторов и компонентов сбора данных
    env = CodeCollabEnv(num_agents=2, max_turns=5)

    actor_optimizer = torch.optim.AdamW(actor_model.parameters(), lr=1e-4)

    actor_optimizers_dict = {
        "agent_0": actor_optimizer,
        "agent_1": actor_optimizer,  # Указываем один оптимизатор, т.к. агенты делят веса LLM
    }
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=3e-4)

    config = SimpleNamespace(num_epochs=10, batch_size=4)
    dataloader = [{"task": "Write a python function to sort a list", "unit_tests": []}]

    collector = RolloutCollector(env, agents_group, critic)
    estimator = GAEEstimator(gamma=0.99, lambda_=0.95)
    buffer = ReplayBuffer(capacity=4)

    actors_dict = {f"agent_{i}": actor for i, actor in enumerate(agents_group.actors)}
    updater = MAACUpdater(
        actors_dict, critic, actor_optimizers_dict, critic_optimizer, MAACLoss()
    )
    
    logger = ConsoleLogger()

    # 5. Запуск цикла обучения
    trainer = MAACTrainer(config, dataloader, collector, estimator, buffer, updater, logger)
    trainer.train()

    print("Обучение завершено!")

    # 6. ОЦЕНКА (Evaluation)
    # Оценка проводится в обычном цикле взаимодействия со средой (без обновления весов)
    print("\n--- Запуск оценки обученных агентов ---")
    test_item = {"task": "Write a python script to ping a server.", "unit_tests": []}
    obs = env.reset(test_item)

    done = False
    total_reward = 0

    while not done:
        # Агенты генерируют действия на основе своих локальных наблюдений
        actions_list = agents_group.act(list(obs.values()))

        # Преобразуем List[AgentAction] обратно в Dict[str, AgentAction]
        actions_dict = {f"agent_{i}": act for i, act in enumerate(actions_list)}

        step_result = env.step(actions_dict)
        obs = step_result.next_observations
        total_reward += step_result.reward
        done = step_result.done

        # Логируем, что написали агенты
        print(f"Turn {env.current_turn}:")
        for a_id, act in actions_dict.items():
            print(f"  {a_id}: {act.text.strip()}")

    print(f"Итоговая награда за эпизод: {total_reward}")


if __name__ == "__main__":
    main()
