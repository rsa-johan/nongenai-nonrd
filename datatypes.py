from enum import Enum
from dataclasses import dataclass
from torch.nn import Module as Neuron

class Optimizers(Enum):
    ADAM = "adam"
    ADA = "ada"
    SGD = "sgd"

class Losses(Enum):
    L1 = "l1"
    L2 = "l2"
    BCELOGITS = "bcelogits"
    BCE = "bce"

@dataclass
class Setup:
    model: Neuron
    loss: Losses = Losses.L1
    optimizer: Optimizers = Optimizers.SGD
    epochs: int = 100
    lr: float = 0.01
    cuda: bool = False
    show_progress: bool = True

