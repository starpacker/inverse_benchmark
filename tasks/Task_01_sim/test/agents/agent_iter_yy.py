import numpy as np


# --- Extracted Dependencies ---

def forward_diff(data, step, dim):
    assert dim <= 2
    r, n, m = np.shape(data)
    size = np.array((r, n, m))
    position = np.zeros(3, dtype='float32')
    temp1 = np.zeros(size + 1, dtype='float32')
    temp2 = np.zeros(size + 1, dtype='float32')

    size[dim] = size[dim] + 1
    position[dim] = position[dim] + 1

    temp1[
        int(position[0]):int(size[0]),
        int(position[1]):int(size[1]),
        int(position[2]):int(size[2])
    ] = data
    temp2[
        int(position[0]):int(size[0]),
        int(position[1]):int(size[1]),
        int(position[2]):int(size[2])
    ] = data

    size[dim] = size[dim] - 1
    temp2[0:size[0], 0:size[1], 0:size[2]] = data
    temp1 = (temp1 - temp2) / step
    size[dim] = size[dim] + 1

    out = temp1[
        int(position[0]):int(size[0]),
        int(position[1]):int(size[1]),
        int(position[2]):int(size[2])]
    return -out

def back_diff(data, step, dim):
    assert dim <= 2
    r, n, m = np.shape(data)
    size = np.array((r, n, m))
    position = np.zeros(3, dtype='float32')
    temp1 = np.zeros(size + 1, dtype='float32')
    temp2 = np.zeros(size + 1, dtype='float32')

    temp1[
        int(position[0]):int(size[0]),
        int(position[1]):int(size[1]),
        int(position[2]):int(size[2])
    ] = data
    temp2[
        int(position[0]):int(size[0]),
        int(position[1]):int(size[1]),
        int(position[2]):int(size[2])
    ] = data

    size[dim] = size[dim] + 1
    position[dim] = position[dim] + 1

    temp2[
        int(position[0]):int(size[0]),
        int(position[1]):int(size[1]),
        int(position[2]):int(size[2])
    ] = data

    temp1 = (temp1 - temp2) / step
    size[dim] = size[dim] - 1
    out = temp1[0:size[0], 0:size[1], 0:size[2]]
    return out

def shrink(x, L):
    s = np.abs(x)
    xs = np.sign(x) * np.maximum(s - 1 / L, 0)
    return xs

def iter_yy(g, byy, para, mu):
    gyy = back_diff(forward_diff(g, 1, 2), 1, 2)
    dyy = shrink(gyy + byy, mu)
    byy = byy + (gyy - dyy)
    Lyy = para * back_diff(forward_diff(dyy - byy, 1, 2), 1, 2)
    return Lyy, byy
