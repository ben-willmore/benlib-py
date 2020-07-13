'''
Gain model estimation and display functions
'''

# pylint: disable=C0103, R0912, R0914

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize, check_grad
from sklearn.base import RegressorMixin
from benlib.strf import concatenate_segments
from benlib.plot import bindata
from benlib.utils import calc_CC_norm
from benlib.lnmodel import sigmoid, estimate_sigmoid

class GainModel(RegressorMixin):
    '''
    Scikit-learn compatible gain model.
    Inputs:
        X[0]: x_t -- usually the output of an STRF model
        X[1]: c_t -- binary valued contrast over time, 0 = low; 1 = high
        y: y_t or y_td -- neuronal response over time or time x trial
    If your data are discontinuous, provide X[0] (etc) as a list of
    continuous segments [X_t0, X_t1, ...]
    Parameters:
        The model has parameters [a, b, c_lo, c_hi, d_lo, d_hi]
        See lnmodel.sigmoid() for the interpretation.
    '''
    def __init__(self):
        self.fit_params = None
        self.fit_result = None
        self.fit_data = None
        self.guess = None

    def fit(self, X=None, y=None):
        '''
        Reshape data (if needed)
        and fit. No tensorization needed because X does not contain history
        '''
        x_t = concatenate_segments(X[0])
        c_t = concatenate_segments(X[1])
        y_t = concatenate_segments(y, mean=True)

        # get a starting guess by roughly estimating
        # sigmoid parameters from data
        w = np.where(c_t == 0)
        bin_x, bin_y = bindata(x_t[w], y_t[w], n_bins=50)
        guess_lo = estimate_sigmoid(bin_x, bin_y)

        w = np.where(c_t == 1)
        bin_x, bin_y = bindata(x_t[w], y_t[w], n_bins=50)
        guess_hi = estimate_sigmoid(bin_x, bin_y)
        self.guess = np.array([(guess_lo[0]+guess_hi[0])/2, (guess_lo[1]+guess_hi[1])/2,
                               guess_lo[2], guess_hi[2], guess_lo[3], guess_hi[3]])

        self.fit_result = minimize(gainmodel_sse, self.guess, args=(x_t, c_t, y_t),
                                   jac=gainmodel_sse_grad, method='cg')
        # print(self.fit_result)
        self.fit_params = self.fit_result.x
        self.fit_data = (x_t, c_t, y_t)

    def predict(self, X=None):
        '''
        Reshape data (if needed)
        and predict
        '''
        x_t = concatenate_segments(X[0])
        c_t = concatenate_segments(X[1])
        return gainmodel(self.fit_params, x_t, c_t)

    def score(self, X=None, y=None, sample_weight=None):
        '''
        Score
        '''
        y = concatenate_segments(y)
        y_hat = self.predict(X)

        if len(y.shape) == 1:
            return np.corrcoef(y, y_hat)

        return calc_CC_norm(y, y_hat)

    def show(self, show_starting_guess=False):
        '''
        Plot sigmoid relationships
        '''
        x_t, c_t, y_t = self.fit_data
        w = np.where(c_t == 0)
        bin_x, bin_y = bindata(x_t[w], y_t[w], n_bins=50)
        params_lo = self.fit_params[np.r_[:3, 4]]
        x_lo = np.linspace(x_t[w].min(), x_t[w].max(), 40)
        plt.scatter(bin_x, bin_y)

        w = np.where(c_t == 1)
        bin_x, bin_y = bindata(x_t[w], y_t[w], n_bins=50)
        params_hi = self.fit_params[np.r_[:2, 3, 5]]
        x_hi = np.linspace(x_t[w].min(), x_t[w].max(), 40)
        plt.scatter(bin_x, bin_y)

        if show_starting_guess:
            plt.plot(x_lo, sigmoid(self.guess[np.r_[:3, 4]], x_lo), 'c')
            plt.plot(x_hi, sigmoid(self.guess[np.r_[:2, 3, 5]], x_hi), 'm')

        plt.plot(x_lo, sigmoid(params_lo, x_lo), 'b')
        plt.plot(x_hi, sigmoid(params_hi, x_hi), 'r')

def gainmodel(params, x_t, c_t):
    '''Gain model with params a, b, c_lo, c_hi, d_lo, d_hi
    '''
    fx = sigmoid(params[np.r_[:3, 4]], x_t)
    w = np.where(c_t == 1)
    fx[w] = sigmoid(params[np.r_[:2, 3, 5]], x_t[w])
    return fx

def gainmodel_sse(params, x_t, c_t, y_t):
    '''
    Gain model SSE loss
    '''
    fx = gainmodel(params, x_t, c_t)
    residuals = fx - y_t
    E = np.sum(residuals**2)
    return E

def gainmodel_sse_grad(params, x_t, c_t, y_t):
    '''
    Jacobian of gain model SSE loss
    '''
    # a is a constant offset therefore doesn't feature in derivative
    _, b, c_lo, c_hi, d_lo, d_hi = params
    fx = gainmodel(params, x_t, c_t)

    residuals = fx - y_t

    # exp(-(z_t-c)/d) recurs in partials
    r_lo = -(x_t-c_lo)/d_lo
    exp_r_lo = np.exp(r_lo)

    r_hi = -(x_t-c_hi)/d_hi
    exp_r_hi = np.exp(r_hi)

    # etrat_sq is a stable approximation to exp(x)/(1+exp(x)^2
    etrat_sq_lo = np.exp(-np.abs(r_lo))/((1+np.exp(-np.abs(r_lo)))**2)
    etrat_sq_hi = np.exp(-np.abs(r_hi))/((1+np.exp(-np.abs(r_hi)))**2)

    dE = np.zeros(6)

    # dfx/da = 1
    dE[0] = np.sum(2*residuals)

    # dfx/db = 1/(1+exp_r)
    dE[1] = np.sum(2*residuals / (1+((c_t == 0)*exp_r_lo)+(c_t == 1)*exp_r_hi))

    # dfx/dc = -b * exp_r /[(1+exp_r)**2 * d]
    #        = -b * etrat / d
    dE[2] = np.sum(2*residuals * -b * (c_t == 0) * etrat_sq_lo / d_lo)
    dE[3] = np.sum(2*residuals * -b * (c_t == 1) * etrat_sq_hi / d_hi)

    # dfx/dd = b * (-x_t+c) * exp_r / [(1+exp_r)**2 * d**2]
    #        = b * (-x_t+c) * etrat / (d**2)
    dE[4] = np.sum(2*residuals * b * (c_t == 0) * (-x_t+c_lo) * etrat_sq_lo / (d_lo**2))
    dE[5] = np.sum(2*residuals * b * (c_t == 1) * (-x_t+c_hi) * etrat_sq_hi / (d_hi**2))

    return dE

def check_gainmodel_grad():
    '''
    Check gainmodel SSE Jacobian
    '''
    err = np.zeros(100)
    for i in range(100):
        p = np.random.random(6)*10
        x = np.random.random(100)
        c = np.random.randint(2, size=(100))
        y = np.random.random(100)
        err[i] = check_grad(gainmodel_sse, gainmodel_sse_grad, p, x, c, y)
    print('Maximum error = %0.3e' % np.max(err))

class GainModel3Free(RegressorMixin):
    '''
    Scikit-learn compatible gain model.
    Inputs:
        X[0]: x_t -- usually the output of an STRF model
        X[1]: c_t -- binary valued contrast over time, 0 = low; 1 = high
        y: y_t or y_td -- neuronal response over time or time x trial
    If your data are discontinuous, provide X[0] (etc) as a list of
    continuous segments [X_t0, X_t1, ...]
    Parameters:
        The model has parameters [a_lo, a_hi, b, c_lo, c_hi, d_lo, d_hi]
        See lnmodel.sigmoid() for the interpretation.
    '''
    def __init__(self):
        self.fit_params = None
        self.fit_result = None
        self.fit_data = None
        self.guess = None

    def fit(self, X=None, y=None):
        '''
        Reshape data (if needed)
        and fit. No tensorization needed because X doesn't contain history
        '''
        x_t = concatenate_segments(X[0])
        c_t = concatenate_segments(X[1])
        y_t = concatenate_segments(y, mean=True)

        # get a starting guess by roughly estimating
        # sigmoid parameters from data
        w = np.where(c_t == 0)
        bin_x, bin_y = bindata(x_t[w], y_t[w], n_bins=50)
        guess_lo = estimate_sigmoid(bin_x, bin_y)

        w = np.where(c_t == 1)
        bin_x, bin_y = bindata(x_t[w], y_t[w], n_bins=50)
        guess_hi = estimate_sigmoid(bin_x, bin_y)

        self.guess = np.array([guess_lo[0], guess_hi[0], (guess_lo[1]+guess_hi[1])/2,
                               guess_lo[2], guess_hi[2], guess_lo[3], guess_hi[3]])

        self.fit_result = minimize(gainmodel_3free_sse, self.guess, args=(x_t, c_t, y_t),
                                   jac=gainmodel_3free_sse_grad, method='cg')

        self.fit_params = self.fit_result.x
        self.fit_data = (x_t, c_t, y_t)

    def predict(self, X=None):
        '''
        Reshape data (if needed)
        and predict
        '''
        x_t = concatenate_segments(X[0])
        c_t = concatenate_segments(X[1])
        return gainmodel_3free(self.fit_params, x_t, c_t)

    def score(self, X=None, y=None, sample_weight=None):
        '''
        Score
        '''
        y = concatenate_segments(y)
        y_hat = self.predict(X)

        if len(y.shape) == 1:
            return np.corrcoef(y, y_hat)

        return calc_CC_norm(y, y_hat)

    def show(self, show_starting_guess=False):
        '''
        Plot sigmoid relationships
        '''
        x_t, c_t, y_t = self.fit_data
        w = np.where(c_t == 0)
        bin_x, bin_y = bindata(x_t[w], y_t[w], n_bins=50)
        params_lo = self.fit_params[np.r_[0, 2:4, 5]]
        x_lo = np.linspace(x_t[w].min(), x_t[w].max(), 40)
        plt.scatter(bin_x, bin_y)

        w = np.where(c_t == 1)
        bin_x, bin_y = bindata(x_t[w], y_t[w], n_bins=50)
        params_hi = self.fit_params[np.r_[1:3, 4, 6]]
        x_hi = np.linspace(x_t[w].min(), x_t[w].max(), 40)
        plt.scatter(bin_x, bin_y)

        if show_starting_guess:
            plt.plot(x_lo, sigmoid(self.guess[np.r_[0, 2:4, 5]], x_lo), 'c')
            plt.plot(x_hi, sigmoid(self.guess[np.r_[1:3, 4, 6]], x_hi), 'm')

        plt.plot(x_lo, sigmoid(params_lo, x_lo), 'b')
        plt.plot(x_hi, sigmoid(params_hi, x_hi), 'r')

def gainmodel_3free(params, x_t, c_t):
    '''Gain model with params a_lo, a_hi, b, c_lo, c_hi, d_lo, d_hi
    '''
    fx = sigmoid(params[np.r_[0, 2:4, 5]], x_t)
    w = np.where(c_t == 1)
    fx[w] = sigmoid(params[np.r_[1:3, 4, 6]], x_t[w])
    return fx

def gainmodel_3free_sse(params, x_t, c_t, y_t):
    '''
    Gain model SSE loss
    '''
    fx = gainmodel_3free(params, x_t, c_t)
    residuals = fx - y_t
    E = np.sum(residuals**2)
    return E

def gainmodel_3free_sse_grad(params, x_t, c_t, y_t):
    '''
    Jacobian of gain model SSE loss
    '''
    # a is a constant offset therefore doesn't feature in derivative
    _, _, b, c_lo, c_hi, d_lo, d_hi = params
    fx = gainmodel_3free(params, x_t, c_t)

    residuals = fx - y_t

    # exp(-(z_t-c)/d) recurs in partials
    r_lo = -(x_t-c_lo)/d_lo
    exp_r_lo = np.exp(r_lo)

    r_hi = -(x_t-c_hi)/d_hi
    exp_r_hi = np.exp(r_hi)

    # etrat_sq is a stable approximation to exp(x)/(1+exp(x)^2
    etrat_sq_lo = np.exp(-np.abs(r_lo))/((1+np.exp(-np.abs(r_lo)))**2)
    etrat_sq_hi = np.exp(-np.abs(r_hi))/((1+np.exp(-np.abs(r_hi)))**2)

    dE = np.zeros(7)

    # dfx/da = 1
    dE[0] = np.sum(2*residuals * (c_t == 0))
    dE[1] = np.sum(2*residuals * (c_t == 1))

    # dfx/db = 1/(1+exp_r)
    dE[2] = np.sum(2*residuals / (1+((c_t == 0)*exp_r_lo)+(c_t == 1)*exp_r_hi))

    # dfx/dc = -b * exp_r /[(1+exp_r)**2 * d]
    #        = -b * etrat / d
    dE[3] = np.sum(2*residuals * -b * (c_t == 0) * etrat_sq_lo / d_lo)
    dE[4] = np.sum(2*residuals * -b * (c_t == 1) * etrat_sq_hi / d_hi)

    # dfx/dd = b * (-x_t+c) * exp_r / [(1+exp_r)**2 * d**2]
    #        = b * (-x_t+c) * etrat / (d**2)
    dE[5] = np.sum(2*residuals * b * (c_t == 0) * (-x_t+c_lo) * etrat_sq_lo / (d_lo**2))
    dE[6] = np.sum(2*residuals * b * (c_t == 1) * (-x_t+c_hi) * etrat_sq_hi / (d_hi**2))

    return dE

def check_gainmodel_3free_grad():
    '''
    Check gainmodel SSE Jacobian
    '''
    err = np.zeros(100)
    for i in range(100):
        p = np.random.random(7)*10
        x = np.random.random(100)
        c = np.random.randint(2, size=(100))
        y = np.random.random(100)
        err[i] = check_grad(gainmodel_3free_sse, gainmodel_3free_sse_grad, p, x, c, y)
    print('Maximum error = %0.3e' % np.max(err))
