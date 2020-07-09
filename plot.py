'''
Plotting functions
'''

# pylint: disable=C0103, R0912, R0914

import numpy as np

def bindata(x, y, n_bins=100):
    '''
    Bin 2D data into n bins, e.g. for plotting sigmoid fits
    '''
    data = np.stack((x, y), axis=-1)
    sort_idx = np.argsort(x)
    data = data[sort_idx, :]

    n = data.shape[0]
    binsize = int(np.ceil(n / n_bins))

    binned = np.zeros((n_bins, 2))
    start_idxs = np.arange(0, n, binsize)
    for i, idx in enumerate(start_idxs):
        binned[i, :] = [np.mean(data[idx:idx+binsize, 0]), np.mean(data[idx:idx+binsize, 1])]
    return binned[:, 0], binned[:, 1]
