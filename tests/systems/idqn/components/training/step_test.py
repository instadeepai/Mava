from types import SimpleNamespace

from mava.systems.idqn.components.training.step import IDQNStep
from mava.systems.trainer import Trainer


class MockTrainer(Trainer):
    def __init__(self) -> None:
        """Init"""
        self.store = SimpleNamespace()


def test_idqn_loss() -> None:
    """Tests that idqn step creates the step function"""
    trainer = MockTrainer()
    loss = IDQNStep()
    loss.on_training_step_fn(trainer)

    assert hasattr(trainer.store, "step_fn")
    assert callable(trainer.store.step_fn)
