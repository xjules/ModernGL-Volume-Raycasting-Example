import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
from numba import cuda, float32
from numba import guvectorize

@cuda.jit(device=True)
def get_median_item(a, N):
    b = cuda.local.array(5, float32)
    for i in range(N):
        b[i] = a[i]

    for end in range(N, 1, -1):
        for i in range(end - 1):
            if b[i] > b[i + 1]:
                tmp = b[i]
                b[i] = b[i+1]
                b[i+1] = tmp

    return b[int(N / 2)]

@cuda.jit('(float32[:,:,:], float32[:,:,:])')
def median_filter_cuda(vol_in, vol_out):
    # idy, idz = cuda.grid(2)
    idx = cuda.threadIdx.z
    idy = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    idz = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z


    check = (idx < vol_in.shape[0] and
             idy < vol_in.shape[1] and
             idz < vol_in.shape[2])

    if check:
        min_z = max(idz - 2, 0)
        max_z = min(idz + 2, vol_out.shape[2])
        b = cuda.local.array(5, float32)

        for i in range(min_z, max_z+1):
            b[i-min_z] = vol_in[idx, idy, i]

        vol_out[idx, idy, idz] = get_median_item(b, max_z - min_z + 1)


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
    return medfilt(arr, (1, 1, 3))

def load_segy():
    import segyio
    filename = os.path.abspath('../gpu_computing/data/full_size/01NmoUpd_8-16stkEps_985_1281.sgy')
    # filename = os.path.abspath('../gpu_computing/data/full_size/relAI-0.sgy')
    # filename = '/data/workspace/graphics_python/gpu_computing/data/01NmoUpd_8-16stkEps_985_1281-cropped.sgy'
    f = segyio.open(filename, iline=5, xline=21)
    data_vol = segyio.tools.cube(f)
    return data_vol


if __name__ == '__main__':
    data = load_segy()
    # fig = plt.figure(figsize=(14, 6))

    data_out = np.empty_like(data)

    stream = cuda.stream()

    threadsperblock = (data.shape[0], 16, 16)
    blockspergrid = (data.shape[1]//threadsperblock[1] + 1, data.shape[2]//threadsperblock[2] + 1)

    print(threadsperblock, blockspergrid)

    with stream.auto_synchronize():
        dA = cuda.to_device(data, stream)
        dB = cuda.to_device(data_out, stream)
        median_filter_cuda[blockspergrid, threadsperblock, stream](dA, dB)
        dB.copy_to_host(data_out, stream)

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    amp1 = ax1.imshow(data[:, 200, :].T, cmap='viridis', aspect='auto')

    amp2 = ax2.imshow(data_out[:, 200, :].T, cmap='viridis', aspect='auto')
    # fig.colorbar(amp1, ax=ax1)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.invert_xaxis()

    plt.show()



