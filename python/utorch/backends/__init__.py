"""Exports different implementations of NDArray."""

import numpy as np

# TODO: https://github.com/semin-park/utorch/issues/1
# In the future, we want our own NDArray implementation for different backends,
# but for now, we're using numpy as our NDArray backend.
NDArray = np.ndarray
