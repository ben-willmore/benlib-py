'''
General utility functions
'''

# pylint: disable=C0103, R0912, R0914

from datetime import datetime
import numpy as np
import scipy.io as spio


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries

    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def sahani_quick(y_td):
    # data must be time x repeats
    n_t, n_d = y_td.shape

    total_power = np.mean(np.var(y_td, axis=0, ddof=1))
    signal_power = 1/(n_d-1) * (n_d * np.var(np.mean(y_td, axis=1), ddof=1) - total_power)
    if signal_power < 0:
        return np.nan, np.nan, total_power
    noise_power = total_power - signal_power
    return signal_power, noise_power, total_power

def calc_CC_norm(y_td, y_hat):
    '''
    Calculate CC_norm, CC_abs_CC_max of a y_td matrix where t is time and d are repeats
    '''
    n_t, n = y_td.shape
    y = np.mean(y_td, axis=1)
    Ey = np.mean(y)
    Eyhat = np.mean(y_hat)
    Vy = np.sum(np.multiply((y-Ey), (y-Ey)))/n_t
    Vyhat = np.sum(np.multiply((y_hat-Eyhat), (y_hat-Eyhat)))/n_t
    Cyyhat = np.sum(np.multiply((y-Ey), (y_hat-Eyhat)))/n_t
    SP = (np.var(np.sum(y_td, axis=1), ddof=1)-np.sum(np.var(y_td, axis=0, ddof=1)))/(n*(n-1))
    CCabs = Cyyhat/np.sqrt(Vy*Vyhat)
    CCnorm = Cyyhat/np.sqrt(SP*Vyhat)
    CCmax = np.sqrt(SP/Vy)
    if SP <= 0:
        print('SP less than or equal to zero - CCmax and CCnorm cannot be calculated.')
        CCnorm = np.nan
        CCmax = 0
    return CCnorm, CCabs, CCmax

class Progress():
    '''
    Progress bar. By default, prints output to screen and file './progress.log'
    '''

    def __init__(self, n, name='Progress', file='./progress.log'):
        self.file = file
        s = '== %s (n=%d)\n%s : start' % \
            (name, n, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        if self.file:
            with open(self.file, 'w') as f:
                f.write(s+'\n')
        print(s)

        self.n = n
        self.i = 0

    def print(self, i=None):
        '''
        Print progress. If i is provided, use it, otherwise increment
        internal counter
        '''
        if i is None:
            i = self.i
            self.i = self.i + 1
        s = '%s : %d of %d complete' % \
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i+1, self.n)
        if self.file:
            with open(self.file, 'a') as f:
                f.write(s+'\n')
        print(s)
