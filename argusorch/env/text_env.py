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


class MultiAgentTextEnv:
    def __init__(self, num_agents: int, max_turns: int) -> None:
        self.num_agents = num_agents
        self.max_turns = max_turns
        self.current_turn = 0
        self.item: Dict[str, Any] | None = None
        self.history: List[Dict[str, Any]] = []

    def reset(self, item: Dict[str, Any]) -> Dict[str, AgentObservation]:
        self.item = item
        self.current_turn = 0
        self.history = []
        return self._generate_observations()

    def step(self, actions: Dict[str, AgentAction]) -> EnvStep:
        self.history.append({
            "turn": self.current_turn,
            "actions": {a_id: act.text for a_id, act in actions.items()},
        })
        # Increment before reward_fn so is_final and done share the same counter.
        self.current_turn += 1
        done = self.current_turn >= self.max_turns

        per_agent_rewards = self.reward_fn(self.item, actions, self.history, done)
        shared_reward = (
            sum(per_agent_rewards.values()) / len(per_agent_rewards)
            if per_agent_rewards
            else 0.0
        )
        next_observations = {} if done else self._generate_observations()

        return EnvStep(
            next_observations=next_observations,
            reward=shared_reward,
            done=done,
            per_agent_rewards=per_agent_rewards,
            info={},
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

    def reward_fn(
        self,
        item: dict,
        actions: dict,
        history: list,
        is_final: bool,
    ) -> Dict[str, float]:
        raise NotImplementedError

    def transition_fn(self, agent_id: str, item: dict, history: list) -> str:
        raise NotImplementedError


_TESTER_NEGATIVE = frozenset({"error", "fail", "wrong", "bug", "broken"})


class CodeCollabEnv(MultiAgentTextEnv):
    def __init__(self, max_turns: int) -> None:
        super().__init__(num_agents=2, max_turns=max_turns)

    def transition_fn(self, agent_id: str, item: dict, history: list) -> str:
        task = item["task"]

        recent = history[-2:] if history else []
        hist_lines: List[str] = []
        for turn in recent:
            for a_id, text in turn["actions"].items():
                role = "Coder" if a_id == "agent_0" else "Tester"
                hist_lines.append(f"{role}: {text[:80]}")
        hist_str = "\n".join(hist_lines) if hist_lines else "(start)"

        if agent_id == "agent_0":
            return (
                f"Task: {task}\n"
                f"History:\n{hist_str}\n"
                "Coder: write a short Python solution."
            )
        else:
            last_coder = ""
            for turn in reversed(history):
                if "agent_0" in turn["actions"]:
                    last_coder = turn["actions"]["agent_0"][:120]
                    break
            return (
                f"Task: {task}\n"
                f"Code:\n{last_coder}\n"
                "Tester: does this code solve the task?\n"
                "Reply PASS (upper-case) if correct, FAIL if not."
            )

    def reward_fn(
        self,
        item: dict,
        actions: dict,
        history: list,
        is_final: bool,
    ) -> Dict[str, float]:
        coder_act = actions.get("agent_0")
        tester_act = actions.get("agent_1")

        coder_text = coder_act.text.strip() if coder_act else ""
        tester_text = tester_act.text.strip() if tester_act else ""
        tester_lower = tester_text.lower()

        # Upper-case "PASS" avoids collision with Python's `pass` keyword in coder output.
        tester_passed = "PASS" in tester_text
        # Word-boundary padding avoids "notable" matching "not".
        tester_negative = any(
            f" {w} " in f" {tester_lower} " for w in _TESTER_NEGATIVE
        ) or "fail" in tester_lower

        if tester_passed:
            coder_reward = 1.0
        elif tester_negative:
            coder_reward = -0.1
        elif any(kw in coder_text for kw in _CODE_KEYWORDS):
            coder_reward = 0.2
        else:
            coder_reward = 0.0

        if tester_passed or tester_negative:
            tester_reward = 0.3
        elif len(tester_text) > 20:
            tester_reward = 0.1
        else:
            tester_reward = -0.05

        return {"agent_0": coder_reward, "agent_1": tester_reward}


_CONTRADICTION_MARKERS = ("contradicts", "contradiction", "however instead", "but actually")


class LongHorizonPlanningEnv(MultiAgentTextEnv):
    _MIN_PLAN_WORDS = 50
    _STEP_MIN_CHARS = 20

    def __init__(self, max_turns: int) -> None:
        super().__init__(num_agents=2, max_turns=max_turns)

    def transition_fn(self, agent_id: str, item: dict, history: list) -> str:
        goal = item["goal"]
        constraints = item.get("constraints", [])
        constr_str = "; ".join(constraints) if constraints else "none"

        recent = history[-3:]
        ctx_lines: List[str] = []
        for h in recent:
            t = h.get("turn", "?")
            for a_id, text in h["actions"].items():
                role = "Planner" if a_id == "agent_0" else "Critic"
                ctx_lines.append(f"[T{t}][{role}]: {text[:80]}")
        ctx = "\n".join(ctx_lines) if ctx_lines else "(start)"

        turn_label = f"{self.current_turn + 1}/{self.max_turns}"

        if agent_id == "agent_0":
            return (
                f"Goal: {goal}\n"
                f"Constraints: {constr_str}\n"
                f"Turn: {turn_label}\n"
                f"Recent:\n{ctx}\n"
                "Planner: propose the next concrete, actionable step."
            )
        else:
            last_planner = ""
            for turn in reversed(history):
                if "agent_0" in turn["actions"]:
                    last_planner = turn["actions"]["agent_0"]
                    break
            return (
                f"Goal: {goal}\n"
                f"Constraints: {constr_str}\n"
                f"Turn: {turn_label}\n"
                f"Proposed step:\n{last_planner}\n"
                "Critic: is this step feasible and consistent with the plan so far?\n"
                "Reply APPROVE or flag issues."
            )

    def reward_fn(
        self,
        item: dict,
        actions: dict,
        history: list,
        is_final: bool,
    ) -> Dict[str, float]:
        planner_act = actions.get("agent_0")
        critic_act = actions.get("agent_1")

        planner_text = planner_act.text.strip() if planner_act else ""
        critic_text = critic_act.text.strip() if critic_act else ""
        critic_lower = critic_text.lower()

        planner_step = 0.1 if len(planner_text) >= self._STEP_MIN_CHARS else -0.05
        critic_approved = "approve" in critic_lower
        critic_flagged_issue = any(m in critic_lower for m in _CONTRADICTION_MARKERS)
        critic_step = 0.1 if (critic_approved or critic_flagged_issue) else 0.0

        if not is_final:
            return {"agent_0": planner_step, "agent_1": critic_step}

        # Final step: aggregate plan quality over the full episode.
        all_planner_text = " ".join(h["actions"].get("agent_0", "") for h in history)
        all_critic_text = " ".join(h["actions"].get("agent_1", "") for h in history)

        plan_is_long = len(all_planner_text.split()) >= self._MIN_PLAN_WORDS
        has_contradiction = any(
            m in all_planner_text.lower() for m in _CONTRADICTION_MARKERS
        ) or any(m in all_critic_text.lower() for m in _CONTRADICTION_MARKERS)

        if plan_is_long and not has_contradiction:
            bonus = 5.0
        elif not has_contradiction:
            bonus = 1.0
        else:
            bonus = -1.0

        return {
            "agent_0": planner_step + bonus,
            "agent_1": critic_step + bonus * 0.5,
        }
