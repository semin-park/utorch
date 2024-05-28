"""Implementations of Node.Op"""

from .autograd import Node
from .backends import NDArray


class ElemWiseAdd(Node.Op):
    """Element-wise addition that takes two Nodes and adds them element-wise."""

    def compute(self, args: list[NDArray]) -> NDArray:
        assert len(args) == 2
        return args[0] + args[1]

    def gradient(self, out_grad: "Node", node: "Node") -> list["Node"]:
        return [out_grad, out_grad]


class ScalarAdd(Node.Op):
    """Scalar addition that takes a Node and adds a constant to each element"""

    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, args: list[NDArray]) -> NDArray:
        assert len(args) == 1
        return self.scalar + args[0]

    def gradient(self, out_grad: "Node", node: "Node") -> list["Node"]:
        return [out_grad]
