import abc
from typing import Dict, Any, Optional

try:
    import wandb
except ImportError:
    wandb = None


class BaseLogger(abc.ABC):
    @abc.abstractmethod
    def log(self, metrics: Dict[str, Any], step: int) -> None:
        pass


class ConsoleLogger(BaseLogger):
    def __init__(self, prefix: str = "Train") -> None:
        self.prefix = prefix

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        metrics_str = " | ".join(
            f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
            for k, v in metrics.items()
        )
        print(f"[{self.prefix}] Step {step} | {metrics_str}")


class WandbLogger(BaseLogger):
    def __init__(
        self,
        project_name: str,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        if wandb is None:
            raise ImportError(
                "wandb is not installed. Please install it using `pip install wandb`."
            )
        self.run = wandb.init(project=project_name, name=run_name, config=config)

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        if self.run is not None:
            wandb.log(metrics, step=step)

    def finish(self) -> None:
        if self.run is not None:
            wandb.finish()
