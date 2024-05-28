"""Implementation of classes that provide auto-differentiation."""

from abc import ABC, abstractmethod
from typing import Optional

from .backends import NDArray


class Node:
    """Base class for Tensor. Represents a node in the computation graph."""

    class Op(ABC):
        """Abstract class that represents an operation."""

        @abstractmethod
        def compute(self, args: list[NDArray]) -> NDArray:
            raise NotImplementedError

        @abstractmethod
        def gradient(self, out_grad: "Node", node: "Node") -> list["Node"]:
            raise NotImplementedError

    op: Optional[Op]
    inputs: list["Node"]
    cached_data: Optional[NDArray]
    requires_grad: bool

    def init(
        self,
        op: Optional[Op],
        inputs: list['Node'],
        cached_data: Optional[NDArray],
        requires_grad: bool = False,
    ):
        self.op = op
        self.inputs = inputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad

    def realize(self) -> NDArray:
        if self.cached_data is None:
            assert self.op is not None
            self.cached_data = self.op.compute([x.realize() for x in self.inputs])
        return self.cached_data
