from typing import Any, Dict, List

from argusorch.env.types import AgentAction, AgentObservation, EnvStep, JointState

_CODE_KEYWORDS = (
    "def ",
    "return ",
    "for ",
    "while ",
    "if ",
    "class ",
    "import ",
)
_POSITIVE_WORDS = (
    "pass",
    "correct",
    "work",
    "ok",
    "done",
    "good",
    "right",
    "success",
    "yes",
    "solv",
)
_NEGATIVE_WORDS = (
    "error",
    "fail",
    "wrong",
    "bug",
    "issue",
    "broken",
    "incorrect",
    "no ",
    "not ",
)


class MultiAgentTextEnv:
    def __init__(self, num_agents: int, max_turns: int) -> None:
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

    def transition_fn(self, agent_id: str, item: dict, history: list) -> str:
        task = item["task"]

        recent = history[-2:] if history else []
        hist_lines = []
        for turn in recent:
            for a_id, text in turn["actions"].items():
                role = "Coder" if a_id == "agent_0" else "Tester"
                hist_lines.append(f"{role}: {text[:80]}")
        hist_str = "\n".join(hist_lines) if hist_lines else "(start)"

        if agent_id == "agent_0":
            return (
                f"Task: {task}\n"
                f"History:\n{hist_str}\n"
                f"Coder: write a short Python solution."
            )
        else:
            last_coder = ""
            for turn in reversed(history):
                if "agent_0" in turn["actions"]:
                    last_coder = turn["actions"]["agent_0"][:120]
                    break
            return (
                f"Task: {task}\n"
                f"Code: {last_coder}\n"
                f"Tester: does this code solve the task? "
                f"Reply PASS if correct, FAIL if not."
            )

    def reward_fn(self, item: dict, actions: dict, history: list) -> float:
        coder_text = actions.get("agent_0", type("", (), {"text": ""})()).text.lower()
        tester_text = actions.get("agent_1", type("", (), {"text": ""})()).text.lower()

        if any(
            w in tester_text
            for w in ("pass", " ok", "correct", "works", "right", "yes", "good", "solv")
        ):
            return 1.0

        if any(
            w in tester_text
            for w in ("error", "fail", "wrong", "bug", "broken", "no ", "not ")
        ):
            return -0.2

        if any(kw in coder_text for kw in _CODE_KEYWORDS):
            return 0.2

        return 0.0


class LongHorizonPlanningEnv(MultiAgentTextEnv):
    _MIN_PLAN_CHARS = 300

    def transition_fn(self, agent_id: str, item: dict, history: list) -> str:
        goal = item["goal"]
        constraints = item.get("constraints", [])
        constr_str = "; ".join(constraints) if constraints else "none"

        recent = history[-3:]
        ctx_lines = []
        for h in recent:
            t = h.get("turn", "?")
            for a_id, text in h["actions"].items():
                ctx_lines.append(f"[T{t}][{a_id}]: {text[:60]}")
        ctx = "\n".join(ctx_lines) if ctx_lines else "(start)"

        return (
            f"Goal: {goal}\n"
            f"Constraints: {constr_str}\n"
            f"Turn: {self.current_turn + 1}/{self.max_turns}\n"
            f"Recent:\n{ctx}\n"
            f"Agent {agent_id}: add the next concrete step."
        )

    def reward_fn(self, item: dict, actions: dict, history: list) -> float:
        is_final = self.current_turn >= self.max_turns - 1

        step_reward = 0.0
        for act in actions.values():
            text = act.text.strip()
            if len(text) > 10:
                step_reward += 0.1
            else:
                step_reward -= 0.05

        if not is_final:
            return step_reward

        full_text = " ".join(str(h["actions"]) for h in history)
        no_contradiction = "contradiction" not in full_text.lower()
        plan_is_long = len(full_text) >= self._MIN_PLAN_CHARS

        if plan_is_long and no_contradiction:
            return step_reward + 5.0
        elif no_contradiction:
            return step_reward + 1.0
        else:
            return step_reward - 1.0
