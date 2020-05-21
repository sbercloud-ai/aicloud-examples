import cudf

import pandas as pd
import time
import cupy as cp

import numpy as np
import warnings

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    # Generate dataset 30000000x3 size on CPU
    rand1 = np.random.randint(low=0, high=int(1e7), size=int(3e7))
    rand2 = np.random.random(size=int(3e7))
    rand3 = np.random.random(size=int(3e7))
    pdf = pd.DataFrame()
    pdf['a'] = rand1
    pdf['b'] = rand2
    pdf['c'] = rand3

    # Generate dataset 30000000x3 size on GPU
    gpu_rand1 = cp.random.randint(low=0, high=int(1e7), size=int(3e7))
    gpu_rand2 = cp.random.random(size=int(3e7))
    gpu_rand3 = cp.random.random(size=int(3e7))

    gdf = cudf.DataFrame()
    gdf['a'] = gpu_rand1
    gdf['b'] = gpu_rand2
    gdf['c'] = gpu_rand3

    del gpu_rand1, gpu_rand2, gpu_rand3, rand1, rand2, rand3

    #
    # Groupby
    #
    start_time = time.time()
    pdf.groupby('a')
    print(f'CPU groupby time = {round(time.time() - start_time, 3)} sec')

    start_time = time.time()
    gdf.groupby('a')
    print(f'GPU groupby time = {round(time.time() - start_time, 3)} sec')

    #
    # Merge
    #
    start_time = time.time()
    pdf = pdf.merge(pdf, on=['a'])
    print(f'CPU merge time = {round(time.time() - start_time, 3)} sec')

    start_time = time.time()
    gdf = gdf.merge(gdf, on=['a'])
    print(f'GPU merge time = {round(time.time() - start_time, 3)} sec')

    #
    # Apply for every value in column["a"]
    #
    def my_udf(x):
        return (x**2) - x

    start_time = time.time()
    pdf['d'] = pdf['a'].apply(my_udf)
    print(f'CPU apply time = {round(time.time() - start_time, 3)} sec')

    start_time = time.time()
    gdf['d'] = gdf['a'].applymap(my_udf)
    print(f'GPU apply time = {round(time.time() - start_time, 3)} sec')

