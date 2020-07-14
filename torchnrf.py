'''
PyTorch model estimation and display functions
'''

# pylint: disable=C0103, R0912, R0914

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.autograd import Variable
from benlib.utils import calc_CC_norm
from benlib.strf import show_strf, tensorize_segments, concatenate_segments

class TorchLinearRegression(torch.nn.Module):
    def __init__(self, n_h=15, learning_rate=1e-2, epochs=2500, lamb=1e-2):
        super(TorchLinearRegression, self).__init__()
        self.n_h = n_h
        self.linear = None
        self.sigmoid = torch.nn.Sigmoid()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lamb = lamb

        self.criterion = torch.nn.MSELoss()
        self.optimizer = None

    def fit(self, X=None, y=None, plot_loss=False):
        X_tfh = tensorize_segments(X, self.n_h)
        n_t, n_f, n_h = X_tfh.shape
        x_train = Variable(torch.from_numpy(X_tfh.reshape(n_t, n_f*n_h)).type(torch.FloatTensor))

        y = concatenate_segments(y, mean=True)
        y_train = Variable(torch.from_numpy(y.reshape(n_t, 1)).type(torch.FloatTensor))

        self.linear = torch.nn.Linear(n_f*n_h, 1, bias=True)
        if torch.cuda.is_available():
            x_train = x_train.cuda()
            y_train = y_train.cuda()
            self.linear = self.linear.cuda()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        saved_loss = np.zeros(self.epochs)
        for epoch in range(self.epochs):
            # clear gradient buffers
            self.optimizer.zero_grad()
            outputs = self.forward(x_train)
            loss = self.criterion(outputs, y_train) + self.lamb*(self.linear.weight.abs()).sum()

            # get gradients w.r.t to parameters
            loss.backward()

            # update parameters
            self.optimizer.step()
            saved_loss[epoch] = loss.item()
            #print('epoch {}, loss {}'.format(epoch, loss.item()))
        if plot_loss:
            plt.plot(np.arange(len(saved_loss)), saved_loss)
        self.kernel = {'k_fh': self.linear.weight.detach().cpu().numpy().reshape(n_f, n_h)}

    def forward(self, x):
        return self.linear(x)

    def predict(self, X=None):
        '''
        Predictions
        '''
        X_tfh = tensorize_segments(X, self.n_h)
        n_t, n_f, n_h = X_tfh.shape
        x = torch.from_numpy(X_tfh.reshape(n_t, n_f*n_h)).type(torch.FloatTensor)

        if torch.cuda.is_available():
            x = x.cuda()
        out = self.forward(x)
        return out.detach().cpu().numpy().ravel()

    def score(self, X=None, y=None, sample_weight=None):
        '''
        Score
        '''
        y = concatenate_segments(y)
        y_hat = self.predict(X)

        if len(y.shape) == 1:
            return np.corrcoef(y, y_hat)[0, 1]

        return calc_CC_norm(y, y_hat)

    def show(self):
        '''
        Show the kernel
        '''

        show_strf(self.kernel['k_fh'])

import torch
from torch.autograd import Variable

class TorchNRF(torch.nn.Module):
    def __init__(self, n_hidden=3, n_h=15, learning_rate=1e-4, epochs=15000, lamb=1e-5):
        super(TorchNRF, self).__init__()
        self.n_h = n_h
        self.linear1 = None
        self.linear2 = None
        self.linear3 = None
        self.n_hidden = n_hidden
        self.sigmoid = torch.nn.Sigmoid()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lamb = lamb

        self.criterion = torch.nn.MSELoss()
        self.optimizer = None

    def fit(self, X=None, y=None, plot_loss=False):
        X_tfh = tensorize_segments(X, self.n_h)
        n_t, n_f, n_h = X_tfh.shape
        x_train = Variable(torch.from_numpy(X_tfh.reshape(n_t, n_f*n_h)).type(torch.FloatTensor))

        y = concatenate_segments(y, mean=True)
        y_train = Variable(torch.from_numpy(y.reshape(n_t, 1)).type(torch.FloatTensor))

        self.linear1 = torch.nn.Linear(n_f*n_h, self.n_hidden, bias=True)
        self.linear2 = torch.nn.Linear(self.n_hidden, 1, bias=True)
        self.linear3 = torch.nn.Linear(1, 1, bias=True)

        if torch.cuda.is_available():
            x_train = x_train.cuda()
            y_train = y_train.cuda()
            self.linear1 = self.linear1.cuda()
            self.linear2 = self.linear2.cuda()
            self.linear3 = self.linear3.cuda()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        saved_loss = np.zeros(self.epochs)
        for epoch in range(self.epochs):

            # clear gradient buffers
            self.optimizer.zero_grad()
            outputs = self.forward(x_train)
            loss = self.criterion(outputs, y_train) + \
                self.lamb*(self.linear1.weight.abs()).sum()
#                 self.lamb*(self.linear2.weight.abs()).sum()
            loss.backward()
            self.optimizer.step()

            saved_loss[epoch] = loss.item()
            #print('epoch {}, loss {}'.format(epoch, loss.item()))

        if plot_loss:
            plt.plot(np.arange(len(saved_loss)), saved_loss)
        self.kernel = {'k_fh': self.linear1.weight.detach().cpu().numpy().reshape(n_f, n_h*self.n_hidden)}

    def forward(self, x):
        out = self.linear3(self.sigmoid(self.linear2(self.sigmoid(self.linear1(x)))))
        return out

    def predict(self, X=None):
        '''
        Predictions
        '''
        X_tfh = tensorize_segments(X, self.n_h)
        n_t, n_f, n_h = X_tfh.shape
        x = torch.from_numpy(X_tfh.reshape(n_t, n_f*n_h)).type(torch.FloatTensor)
        if torch.cuda.is_available():
            x = x.cuda()
        out = self.forward(x)
        return out.detach().cpu().numpy().ravel()

    def score(self, X=None, y=None, sample_weight=None):
        '''
        Score
        '''
        y = concatenate_segments(y)
        y_hat = self.predict(X)

        if len(y.shape) == 1:
            return np.corrcoef(y, y_hat)[0, 1]

        return calc_CC_norm(y, y_hat)

    def show(self):
        '''
        Show the kernel
        '''

        show_strf(self.kernel['k_fh'])
