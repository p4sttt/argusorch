from dataclasses import dataclass
from typing import Any, Union
import torch


@dataclass
class PolicyEval:
    logprobs: Union[float, torch.Tensor]


@dataclass
class ValuePrediction:
    values: Union[float, torch.Tensor]
    critic_prompt: str
