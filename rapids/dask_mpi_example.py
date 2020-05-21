import cudf

import pandas as pd
import time

from dask.distributed import Client, LocalCluster
from dask_cuda import LocalCUDACluster
from dask_mpi import initialize
from mpi4py import MPI
from dask_cuda.utils import get_n_gpus
import dask.dataframe as dd
import dask
import dask_cudf
from numba import jit
import cupy as cp
import os
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def make_cudf_dataframe(nrows=int(1e8)):
    # Comm - see every workers and processes
    comm = MPI.COMM_WORLD
    # gpu_device - concrete GPU on current worker
    gpu_device = comm.rank % get_n_gpus()
    cp.cuda.Device(gpu_device).use()
    cudf_df = cudf.DataFrame()
    cudf_df['a'] = cp.random.randint(low=0, high=1000, size=nrows)
    cudf_df['b'] = cp.random.randint(low=0, high=1000, size=nrows)
    cudf_df['c'] = cp.random.random(nrows)
    cudf_df['d'] = cp.random.random(nrows)
    return cudf_df


if __name__ == "__main__":
    # Increase memory limit for Dask worker to utilize all available memory
    initialize(memory_limit=1073741824000)
    client = Client()

    # Process every part of datatframe on it's own GPU
    # Generation in 10 streams
    delayed_cudf_dataframe = [dask.delayed(make_cudf_dataframe)() for i in range(10)]
    ddf = dask_cudf.from_delayed(delayed_cudf_dataframe)

    start_time = time.time()
    # Start computation
    ddf.groupby(['a', 'b']).agg({"c": ['sum', 'mean']}).compute()
    print(f'GPUs agg time = {round(time.time() - start_time, 2)} sec')








