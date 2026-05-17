class LLMActor:
    def __init__(self, model, tokenizer, generation_config):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config

    def act(self, obs: AgentObservation) -> AgentAction:
        completion = self.generate(obs.prompt)
        logprob = self.compute_logprob(obs.prompt, completion)
        return AgentAction(text=completion, logprob=logprob)

    def evaludate_action(self, obs, act) -> PolicyEval:
        logprob = self.compute_logprob(observation.prompt, action.text)
        return PolicyEval(logprob=logprob)