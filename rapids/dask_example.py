import cudf

import pandas as pd
import time

from dask.distributed import Client, LocalCluster
from dask_cuda import LocalCUDACluster
import dask.dataframe as dd
from dask_cuda.utils import get_n_gpus
import dask
import dask_cudf
from numba import jit
import cupy as cp
import numpy as np
import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    """
    Distributed Preprocessing
    """
    cluster = LocalCUDACluster()
    client = Client(cluster)

    def make_cudf_dataframe(nrows=int(1e8)):
        # Using rapids DataFrame
        cudf_df = cudf.DataFrame()
        cudf_df['a'] = cp.random.randint(low=0, high=1000, size=nrows)
        cudf_df['b'] = cp.random.randint(low=0, high=1000, size=nrows)
        cudf_df['c'] = cp.random.random(nrows)
        cudf_df['d'] = cp.random.random(nrows)
        return cudf_df

    delayed_cudf_dataframe = [dask.delayed(make_cudf_dataframe)() for i in range(20)]
    start_time = time.time()
    ddf = dask_cudf.from_delayed(delayed_cudf_dataframe)
    
    print(f'dataset len = {len(ddf)}')
    ddf.groupby(['a', 'b']).agg({"c": ['sum', 'mean']}).compute()
    print(f'GPUs agg time = {round(time.time() - start_time, 2)} sec')
    client.close()
    cluster.close()
