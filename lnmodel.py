'''
LN model estimation and display functions
'''

# pylint: disable=C0103, R0912, R0914

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize, check_grad
from sklearn.base import RegressorMixin
from benlib.strf import SeparableKernel, ElNet, tensorize_segments, concatenate_segments
from benlib.plot import bindata

class ElNet_Sigmoid(RegressorMixin):
    '''
    ElNet + Sigmoid
    '''
    def __init__(self, n_h=15, l1_ratio=None):
        if not l1_ratio:
            l1_ratio = [0.001, .25, .5, 0.75, 1]
        self.n_h = n_h
        self.elnet = ElNet(l1_ratio=l1_ratio)
        self.sigmoid = Sigmoid()

    def fit(self, X=None, y=None):
        '''
        Reshape data (if needed)
        and fit
        '''
        X = tensorize_segments(X, self.n_h)
        y = concatenate_segments(y)
        self.elnet.fit(X, y)
        pred = self.elnet.predict(X)
        self.sigmoid.fit(pred, y)

    def predict(self, X=None):
        '''
        Reshape data (if needed)
        and predict
        '''
        X = tensorize_segments(X, self.n_h)
        return self.sigmoid.predict(self.elnet.predict(X))

class SepKernel_Sigmoid(RegressorMixin):
    '''
    Separable kernel + Sigmoid
    '''
    def __init__(self, n_h=15, n_iter=15):
        self.n_h = n_h
        self.sepkernel = SeparableKernel(n_h=n_h, n_iter=n_iter)
        self.sigmoid = Sigmoid()

    def fit(self, X=None, y=None):
        '''
        Reshape data (if needed)
        and fit
        '''
        X = tensorize_segments(X, self.n_h)
        y = concatenate_segments(y)
        self.sepkernel.fit(X, y)
        pred = self.sepkernel.predict(X)
        self.sigmoid.fit(pred, y)

    def predict(self, X=None):
        '''
        Reshape data (if needed)
        and predict
        '''
        X = tensorize_segments(X, self.n_h)
        return self.sigmoid.predict(self.sepkernel.predict(X))

class Sigmoid(RegressorMixin):
    '''
    Scikit-learn compatible sigmoid
    '''
    def __init__(self):
        self.fit_params = None
        self.fit_result = None
        self.fit_data = None
        self.guess = None

    def fit(self, X=None, y=None):
        '''
        Reshape data (if needed)
        and fit. No tensoriztion needed because X is 1D here.
        '''
        X = concatenate_segments(X)
        y = concatenate_segments(y)

        # get a starting guess by roughly estimating
        # sigmoid parameters from data
        bin_x, bin_y = bindata(X, y, n_bins=50)
        self.guess = estimate_sigmoid(bin_x, bin_y)

        self.fit_result = minimize(sigmoid_sse, self.guess, args=(X, y),
                                   jac=sigmoid_sse_grad, method='cg')
        # print(self.fit_result)
        self.fit_params = self.fit_result.x
        self.fit_data = (X, y)

    def predict(self, X=None):
        '''
        Reshape data (if needed)
        and predict
        '''
        X = concatenate_segments(X)
        return sigmoid(self.fit_params, X)

    def show(self, show_starting_guess=False):
        '''
        Plot sigmoid relationship
        '''
        x, y = self.fit_data
        bin_x, bin_y = bindata(x, y, n_bins=100)
        plt.scatter(bin_x, bin_y)
        x = np.linspace(x.min(), x.max(), 40)
        if show_starting_guess:
            plt.plot(x, sigmoid(self.guess, x), 'r')
        plt.plot(x, self.predict(x), 'g')

def sigmoid(params, x):
    '''
    Robust sigmoid
    '''
    a, b, c, d = params

    g = (x-c)/d

    w = np.where(g >= 0)
    z = np.exp(-g[w])
    g[w] = 1 / (1 + z)

    w = np.where(g < 0)
    z = np.exp(g[w])
    g[w] = z / (1 + z)

    # looks faster, but isnt:
    # g = np.where(e>=0, np.exp(1 / (1 + np.exp(-e))), np.exp(e) / (1+np.exp(e)))

    return a + b * g

def sigmoid_sse(params, X, y):
    '''
    Sigmoid SSE loss
    '''
    fX = sigmoid(params, X)
    residuals = fX - y
    E = np.sum(residuals**2)
    return E

def sigmoid_sse_grad(params, x, y):
    '''
    Jacobian of sigmoid SSE loss
    '''
    a, b, c, d = params
    fx = sigmoid(params, x)

    residuals = fx - y

    # exp(-(z_t-c)/d) recurs in partials
    r = -(x-c)/d
    exp_r = np.exp(r)

    # etrat_sq is a stable approximation to exp(x)/(1+exp(x)^2
    etrat_sq = np.exp(-np.abs(r))/((1+np.exp(-np.abs(r)))**2)

    dE = np.zeros(4)

    # dfx/da = 1
    dE[0] = np.sum(2*residuals)

    # dfx/db = 1./(1+exp_r)
    dE[1] = np.sum(2*residuals / (1+exp_r))

    # dfx/dc = -b*exp_r /[d*(1+exp_r)^2]
    #        = -b /[d*(1+exp_r)] * etrat
    dE[2] = np.sum(2*residuals * -b * etrat_sq / d)

    # dfx/dd = b(-z_t+c)*exp_r / [d*(1+exp_r)]^2
    #        = b(-z_t+c)/[d^2*(1+exp_r)] * etrat
    dE[3] = np.sum(2*residuals * b * (-x+c) * etrat_sq / (d**2))

    return dE

def check_sigmoid_grad():
    '''
    Check sigmoid SSE Jacobian
    '''
    x = np.array([5, 6, 7, 8])
    y = np.array([1, 2, 3, 4])
    check_grad(sigmoid_sse, sigmoid_sse_grad, np.random.random(4,)*100, x, y)

def estimate_sigmoid(x, y, d_prop=0.05):
    '''
    Roughly estimate a sigmoid fit to data
    '''
    # a is the minimum value of y
    a = np.min(y)

    # b is the range of y (max-min)
    b = np.max(y) - a

    # c is the x value where y crosses the halfway point
    y_d = np.abs(y-(a+0.5*b))
    idx_mn = np.argmin(y_d)
    c = x[idx_mn]

    # d is b / 4g where g is the gradient at the halfway point
    y_hi = y.copy()
    y_hi[y < a+(1-d_prop)*b] = np.inf
    idx_hi = np.argmin(y_hi)
    pt_hi = (x[idx_hi], y[idx_hi])

    y_lo = y.copy()
    y_lo[y > a+d_prop*b] = -np.inf
    idx_lo = np.argmax(y_lo)
    pt_lo = (x[idx_lo], y[idx_lo])

    g = (pt_hi[1]-pt_lo[1])/(pt_hi[0]-pt_lo[0])

    d = b / (4 * g)

    return [a, b, c, d]
