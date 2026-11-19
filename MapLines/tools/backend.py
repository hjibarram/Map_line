
"""
CPU and CUDA backend management for MapLine.

This module provides utilities to select the numerical backend used
for spectral modeling and Bayesian parameter estimation.

NumPy is the default CPU backend, while CuPy provides optional GPU
acceleration through NVIDIA CUDA.

The module is designed to preserve compatibility with systems
without CUDA-compatible hardware or CuPy installations.

Supported execution modes
-------------------------
auto
    Automatically select CUDA when available, otherwise use CPU.

cpu
    Force NumPy execution, regardless of CUDA availability.

cuda
    Require CUDA execution. Raise an exception if CUDA is
    unavailable or cannot be initialized.

Notes
-----
CUDA acceleration requires CuPy and a compatible NVIDIA GPU.
The CUDA Toolkit compiler (nvcc) is not required for standard
CuPy array operations.

Selecting CuPy does not automatically make existing NumPy or
SciPy functions GPU-compatible. Numerical functions must be
implemented using the selected backend.

Examples
--------
>>> xp, use_cuda = get_backend("cpu")
>>> xp.__name__
'numpy'
>>> use_cuda
False

>>> values = xp.asarray([1.0, 2.0, 3.0])
>>> values.shape
(3,)
"""

import sys
import numpy as np


def get_backend(backend="auto"):
    """
    Select the numerical backend for MapLine computations.

    This function determines whether NumPy or CuPy should be used
    for numerical calculations, according to the requested backend
    and the availability of CUDA-compatible hardware.

    In automatic mode, CUDA initialization is attempted first.
    If CuPy is unavailable, no compatible GPU is detected, or CUDA
    initialization fails, NumPy is selected as a fallback.

    Parameters
    ----------
    backend : {'auto', 'cpu', 'cuda'}, optional
        Numerical backend selection mode.

        - 'auto': Select CuPy when CUDA is available; otherwise,
          fall back to NumPy.
        - 'cpu': Always select NumPy.
        - 'cuda': Require CuPy and a functional CUDA device.

        Default is 'auto'.

    Returns
    -------
    xp : module
        Numerical array module corresponding to the selected
        backend: `numpy` or `cupy`.

    use_cuda : bool
        True if CUDA is available and selected.
        False if NumPy is selected.

    Raises
    ------
    ValueError
        If `backend` is not one of 'auto', 'cpu', or 'cuda'.

    RuntimeError
        If CUDA is explicitly requested but CuPy cannot be
        imported, no CUDA device is detected, or GPU initialization
        fails.

    Notes
    -----
    CUDA detection is performed when this function is called,
    rather than when the module is imported.

    A small GPU operation is executed to verify that the CUDA
    runtime can initialize and perform numerical calculations.

    Successful initialization does not guarantee that every
    MapLine model supports CUDA execution.

    Examples
    --------
    Force CPU execution:

    >>> xp, use_cuda = get_backend("cpu")
    >>> print(xp.__name__)
    numpy

    Automatically select the backend:

    >>> xp, use_cuda = get_backend("auto")

    Require CUDA:

    >>> xp, use_cuda = get_backend("cuda")
    """
    if backend not in ("auto", "cpu", "cuda"):
        raise ValueError(
            "Invalid backend '{}'. "
            "Expected 'auto', 'cpu', or 'cuda'.".format(backend)
        )

    if backend == "cpu":
        return np, False

    try:
        import cupy as cp

        ndevices = cp.cuda.runtime.getDeviceCount()

        if ndevices < 1:
            raise RuntimeError(
                "No CUDA-compatible GPU detected."
            )

        # Verify that CUDA can execute a numerical operation.
        test = cp.arange(10, dtype=cp.float32)
        result = cp.sum(test)

        # Synchronize and transfer the scalar to the CPU.
        result.item()

        return cp, True

    except Exception as exc:
        if backend == "cuda":
            raise RuntimeError(
                "CUDA execution was requested, but CUDA "
                "could not be initialized."
            ) from exc

        return np, False


def to_numpy(array):
    """
    Convert an array-like object to a NumPy array.

    This function provides a common interface for transferring
    numerical results from GPU memory to CPU memory.

    CuPy arrays are explicitly transferred from the GPU to the
    host using `cupy.asnumpy`. NumPy arrays and other compatible
    array-like objects are converted using `numpy.asarray`.

    Parameters
    ----------
    array : array_like
        Input numerical data. Supported inputs include NumPy
        arrays, CuPy arrays, lists, tuples, and scalar values.

    Returns
    -------
    numpy.ndarray
        NumPy representation of the input data.

    Notes
    -----
    For NumPy arrays, this function generally avoids unnecessary
    memory copies.

    For CuPy arrays, a device-to-host transfer is performed.
    Frequent transfers between CPU and GPU can significantly
    reduce the performance benefits of CUDA acceleration.

    This function is particularly useful when returning
    GPU-computed likelihood values to CPU-based MCMC samplers
    such as emcee.

    Examples
    --------
    Convert a NumPy array:

    >>> import numpy as np
    >>> a = np.arange(5)
    >>> b = to_numpy(a)
    >>> isinstance(b, np.ndarray)
    True

    Convert a CuPy array:

    >>> import cupy as cp
    >>> a = cp.arange(5)
    >>> b = to_numpy(a)
    >>> isinstance(b, np.ndarray)
    True
    """
    cp = sys.modules.get("cupy")

    if cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)

    return np.asarray(array)


def get_backend_name(xp):
    """
    Return the name of the numerical backend.

    Parameters
    ----------
    xp : module
        Numerical backend module, typically NumPy or CuPy.

    Returns
    -------
    str
        Name of the selected backend, either 'cpu' or 'cuda'.

    Raises
    ------
    ValueError
        If the provided module is not recognized as NumPy
        or CuPy.

    Examples
    --------
    >>> import numpy as np
    >>> get_backend_name(np)
    'cpu'
    """
    if xp is np:
        return "cpu"

    cp = sys.modules.get("cupy")

    if cp is not None and xp is cp:
        return "cuda"

    raise ValueError(
        "Unknown numerical backend module."
    )


def get_device_info(backend="auto"):
    """
    Retrieve information about the selected computing device.

    This function reports the backend being used and, when CUDA
    is available and selected, the name and index of the active
    NVIDIA GPU.

    Parameters
    ----------
    backend : {'auto', 'cpu', 'cuda'}, optional
        Requested backend selection mode.
        Default is 'auto'.

    Returns
    -------
    info : dict
        Dictionary containing:

        - 'backend': Selected backend ('cpu' or 'cuda').
        - 'device': Computing device name.
        - 'device_id': CUDA device index, or None for CPU.
        - 'cuda_enabled': Whether CUDA is active.

    Raises
    ------
    ValueError
        If an invalid backend mode is specified.

    RuntimeError
        If CUDA is explicitly requested but unavailable.

    Examples
    --------
    >>> info = get_device_info("cpu")
    >>> info["backend"]
    'cpu'

    >>> info["cuda_enabled"]
    False
    """
    xp, use_cuda = get_backend(backend)

    if use_cuda:
        device_id = xp.cuda.runtime.getDevice()

        properties = xp.cuda.runtime.getDeviceProperties(
            device_id
        )

        device_name = properties["name"]

        if isinstance(device_name, bytes):
            device_name = device_name.decode("utf-8")

        return {
            "backend": "cuda",
            "device": device_name,
            "device_id": device_id,
            "cuda_enabled": True,
        }

    return {
        "backend": "cpu",
        "device": "CPU (NumPy)",
        "device_id": None,
        "cuda_enabled": False,
    }
