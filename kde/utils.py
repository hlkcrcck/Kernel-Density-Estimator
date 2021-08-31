import numpy as np
import multiprocessing


def make_data(obs, dim, f=0.3, random_seed=1):
    rand = np.random.RandomState(random_seed)
    x = rand.randn(obs, dim)
    x[int(f * obs):] += 5
    return x


def unpacking_apply_along_axis(all_args):
    (func1d, axis, arr, args, kwargs) = all_args
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)


def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple
    cores.
    """
    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):
    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)

    # Chunks for the mapping (only a few chunks):
    chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, multiprocessing.cpu_count())]

    pool = multiprocessing.Pool()
    individual_results = pool.map(unpacking_apply_along_axis, chunks)
    # Freeing the workers:
    pool.close()
    pool.join()

    return np.concatenate(individual_results)


def at_least_1d(data):
    if len(data) == 0:
        raise ValueError("Data is empty!")
    elif len(data.shape) == 2:
        return data
    elif len(data.shape) == 1:
        return data[-1, 1]
    else:
        raise ValueError("Data must be one or two dimensional!")
