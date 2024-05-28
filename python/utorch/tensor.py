"""Implementation of Tensor"""

from typing import Optional, Union

from . import ops
from .autograd import Node
from .backends import NDArray


class Tensor(Node):
    """Tensor is:
    * a multi-dimensional array to the users
    * a node in the computational graph to the framework
    """

    grad: Optional["Tensor"]

    def __init(
        self,
        *,
        op: Optional[Node.Op] = None,
        inputs: Optional[list[Node]] = None,
        cached_data: Optional[NDArray] = None,
        requires_grad: bool = False,
    ):
        self.init(op, inputs or [], cached_data, requires_grad)
        self.grad = None

    @classmethod
    def from_compute(cls, op: Optional[Node.Op], inputs: list[Node]) -> "Tensor":
        requires_grad = any(x.requires_grad for x in inputs)
        tensor = cls.__new__(cls)
        tensor.__init(op=op, inputs=inputs, requires_grad=requires_grad)
        return tensor

    @classmethod
    def from_constant(cls, data: NDArray) -> "Tensor":
        tensor = cls.__new__(cls)
        tensor.__init(cached_data=data)
        return tensor

    def __init__(self, other: Union[NDArray, "Tensor"], *, requires_grad=False):
        if isinstance(other, NDArray):
            self.__init(cached_data=other, requires_grad=requires_grad)
        elif isinstance(other, Tensor):
            self.__init(op=other.op, inputs=other.inputs, requires_grad=requires_grad)
        else:
            raise RuntimeError(f"Unknown conversion to Tensor from: {type(other)}")

    @property
    def shape(self):
        return self.realize().shape

    def __str__(self):
        # TODO: Reimplement after fixing https://github.com/semin-park/utorch/issues/1
        return str(self.realize())

    def __repr__(self):
        # TODO: Reimplement after fixing https://github.com/semin-park/utorch/issues/1
        lines = str(self.realize()).split("\n")
        lines[0] = "Tensor(" + lines[0]
        lines[1:] = ["       " + line for line in lines[1:]]
        lines[-1] += ")"
        return "\n".join(lines)

    def __add__(self, other: Union[float, "Tensor"]):
        if isinstance(other, Tensor):
            assert other.shape == self.shape
            return Tensor.from_compute(ops.ElemWiseAdd(), [self, other])
        return Tensor.from_compute(ops.ScalarAdd(other), [self])
