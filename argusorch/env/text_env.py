from typing import Any, Dict, List

from .types import AgentAction, AgentObservation, EnvStep, JointState


class MultiAgentTextEnv:
    def __init__(self, num_agents: int, max_turns: int):
        self.num_agents = num_agents
        self.max_turns = max_turns
        self.current_turn = 0
        self.item = None

        self.history: List[Dict[str, Any]] = []

    def reset(self, item: Dict[str, Any]) -> Dict[str, AgentObservation]:
        self.item = item
        self.current_turn = 0
        self.history = []

        return self._generate_observations()

    def step(self, actions: Dict[str, AgentAction]) -> EnvStep:
        step_record = {
            "turn": self.current_turn,
            "actions": {a_id: act.text for a_id, act in actions.items()},
        }
        self.history.append(step_record)

        reward = self.reward_fn(self.item, actions, self.history)

        self.current_turn += 1
        done = self.current_turn >= self.max_turns

        next_observations = self._generate_observations()

        return EnvStep(
            next_observations=next_observations, reward=reward, done=done, info={}
        )

    def joint_state(self) -> JointState:
        return JointState(item=self.item, history=self.history, turn=self.current_turn)

    def _generate_observations(self) -> Dict[str, AgentObservation]:
        observations = {}
        for i in range(self.num_agents):
            agent_id = f"agent_{i}"
            prompt = self.transition_fn(agent_id, self.item, self.history)
            observations[agent_id] = AgentObservation(agent_id=agent_id, prompt=prompt)
        return observations

    def reward_fn(self, item: dict, actions: dict, history: list) -> float:
        raise NotImplementedError

    def transition_fn(self, agent_id: str, item: dict, history: list) -> str:
        raise NotImplementedError


class CodeCollabEnv(MultiAgentTextEnv):
    """
    Item содержит: {'task': str, 'unit_tests': list}
    Агенты: agent_0 (Coder), agent_1 (Reviewer/Tester)
    """

    def transition_fn(self, agent_id: str, item: dict, history: list) -> str:
        task = item["task"]
        formatted_history = ""
        for turn in history:
            for a_id, text in turn["actions"].items():
                formatted_history += f"{a_id}: {text}\n"

        if agent_id == "agent_0":
            return f"Task: {task}\nHistory:\n{formatted_history}\nYou are the Coder. Write or fix the code."
        else:
            return f"Task: {task}\nHistory:\n{formatted_history}\nYou are the Tester. Provide feedback and test results."

    def reward_fn(self, item: dict, actions: dict, history: list) -> float:
        # Для симуляции: даем награду, если в ответе Тестера есть 'PASSED'
        last_tester_output = actions.get("agent_1", "").text
        if "PASSED" in last_tester_output:
            return 1.0  # Успех
        elif "ERROR" in last_tester_output:
            return -0.1  # Штраф за ошибку
        return 0.0


class LongHorizonPlanningEnv(MultiAgentTextEnv):
    """
    Item содержит: {'goal': str, 'constraints': list}
    Максимальное количество ходов (max_turns): 10-15
    """

    def transition_fn(self, agent_id: str, item: dict, history: list) -> str:
        goal = item["goal"]
        # Показываем только последние 3 шага, чтобы агенты учились
        # полагаться на сжатое состояние (как в реальных LLM с лимитом контекста)
        last_history = history[-3:]
        context = "\n".join([str(h["actions"]) for h in last_history])

        return (
            f"Global Goal: {goal}\n"
            f"Recent Steps: {context}\n"
            f"Agent {agent_id}, contribute to the next step of the long-term plan."
        )

    def reward_fn(self, item: dict, actions: dict, history: list) -> float:
        # Награда выдается ТОЛЬКО в конце эпизода
        if self.current_turn < self.max_turns - 1:
            return 0.0

        # Симулируем: проверяем длину истории и отсутствие противоречий
        full_text = " ".join([str(h["actions"]) for h in history])
        if len(full_text) > 1000 and "contradiction" not in full_text.lower():
            return 10.0  # Большая награда за длинный и связный план
        return -1.0
