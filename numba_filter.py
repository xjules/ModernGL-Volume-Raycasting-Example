import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
from numba import cuda
from numba import guvectorize

#
# @cuda.jit
# def process_gauss_volume(vol_in, vol_out):
#     idy, idz = cuda.grid(2)
#     idx = cuda.blockIdx.z
#     for i in range(-2, 3):
#         ix = idx + i
#
#         for j in range(-2, 3):
#             for k in range(-2, 3):



# do 1D filtering along z axis!!!!

@guvectorize(['void(float32[:], intp[:], float32[:])'], '(n),()->(n)')
def move_mean(a, window_arr, out):
    window_width = window_arr[0]
    asum = 0.0
    count = 0
    for i in range(window_width):
        asum += a[i]
        count += 1
        out[i] = asum / count
    for i in range(window_width, len(a)):
        asum += a[i] - a[i - window_width]
        out[i] = asum / count

#do @vectorize, @guvectorize, @jit @

#median filter
def median_filter(arr):
    from scipy.signal import medfilt
    return medfilt(arr, (3, 3, 3))

def load_segy():
    import segyio
    filename = os.path.abspath('../gpu_computing/data/full_size/01NmoUpd_8-16stkEps_985_1281.sgy')
    filename = os.path.abspath('../gpu_computing/data/full_size/relAI-0.sgy')
    filename = '/data/workspace/graphics_python/gpu_computing/data/01NmoUpd_8-16stkEps_985_1281-cropped.sgy'
    f = segyio.open(filename, iline=5, xline=21)
    data_vol = segyio.tools.cube(f)
    return data_vol


if __name__ == '__main__':
    data = load_segy()
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(111)
    amp = ax1.imshow(data[:, 200, :].T, cmap='viridis')
    fig.colorbar(amp, ax=ax1)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.invert_xaxis()

    plt.show()


    # griddim = 1, 2
    # blockdim = 3, 4
    # foo[griddim, blockdim](aryA, aryB)



