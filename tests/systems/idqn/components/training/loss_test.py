from types import SimpleNamespace

from mava.systems.idqn.components.training.loss import IDQNLoss
from mava.systems.trainer import Trainer


class MockTrainer(Trainer):
    def __init__(self) -> None:
        """Init"""
        self.store = SimpleNamespace()


def test_idqn_loss() -> None:
    """Tests that idqn loss creates the loss/grad function"""
    trainer = MockTrainer()
    loss = IDQNLoss()
    loss.on_training_loss_fns(trainer)

    assert hasattr(trainer.store, "policy_grad_fn")
    assert callable(trainer.store.policy_grad_fn)
