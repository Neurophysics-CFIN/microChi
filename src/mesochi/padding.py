"""Zero padding after each spatial axis."""
import numpy as np


def padarray_post(x, pad_size, dtype_like=None):
    """Pad with zeros after each axis, optionally using dtype_like's dtype."""
    x = np.asarray(x)
    pad_size = np.asarray(pad_size)
    if (pad_size.shape != (x.ndim,) or not np.all(np.isfinite(pad_size))
            or np.any(pad_size < 0) or np.any(pad_size != np.floor(pad_size))):
        raise ValueError('pad_size must give a nonnegative integer for each axis.')
    if dtype_like is not None:
        x = x.astype(np.asarray(dtype_like).dtype, copy=False)
    return np.pad(x, tuple((0, int(p)) for p in pad_size), mode='constant')
