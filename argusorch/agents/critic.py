class CentralizedCritic:
    def __init__(self, model, tokenizer, prompt_builder):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_builder = prompt_builder

    def evaluate_state(self, joint_state: JointState) -> ValuePrediction:
        critic_prompt = self.prompt_builder.build(joint_state)
        value = self.forward_value(critic_prompt)
        return ValuePrediction(value=value, critic_prompt=critic_prompt)
