from abc import ABC, abstractmethod

from argusorch.env import JointState


class PromptBuilder(ABC):
    @abstractmethod
    def build(self, joint_state: JointState) -> str:
        pass


class DefaultPromptBuilder(PromptBuilder):
    def build(self, joint_state: JointState) -> str:
        return str(joint_state)
