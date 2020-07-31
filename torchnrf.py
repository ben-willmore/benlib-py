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
    def __init__(self, n_h=15, n_fut=0, learning_rate=1e-2, epochs=2500, lamb=1e-2):
        super(TorchLinearRegression, self).__init__()
        self.n_h = n_h
        self.n_fut = n_fut
        self.linear = None
        self.sigmoid = torch.nn.Sigmoid()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lamb = lamb

        self.criterion = torch.nn.MSELoss()
        self.optimizer = None

    def fit(self, X=None, y=None, plot_loss=False):
        X_tfh = tensorize_segments(X, self.n_h, n_fut=self.n_fut)
        n_t, self.n_f, n_hall = X_tfh.shape
        x_train = Variable(torch.from_numpy(X_tfh.reshape(n_t, self.n_f*n_hall)).type(torch.FloatTensor))

        y = concatenate_segments(y, mean=True)
        y_train = Variable(torch.from_numpy(y.reshape(n_t, 1)).type(torch.FloatTensor))

        self.linear = torch.nn.Linear(self.n_f*n_hall, 1, bias=True)
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
            loss = self.criterion(outputs, y_train) + \
                self.lamb*(self.linear.weight.norm(p=1))

            # get gradients w.r.t to parameters
            loss.backward()

            # update parameters
            self.optimizer.step()
            saved_loss[epoch] = loss.item()
            #print('epoch {}, loss {}'.format(epoch, loss.item()))
        if plot_loss:
            plt.plot(np.arange(len(saved_loss)), saved_loss)

        w = self.linear.weight.detach().cpu().numpy()
        self.info = {'type': 'TorchLinearRegression',
                     'n_f': self.n_f,
                     'n_h': self.n_h,
                     'n_fut': self.n_fut,
                     'c': self.linear.bias.detach().cpu().numpy(),
                     'k_fh': w.reshape(self.n_f, n_hall), # for convenience,
                     'b': self.linear.bias.detach().cpu().numpy(),
                     'w': w
                     }

    def forward(self, x):
        return self.linear(x)

    def predict(self, X=None):
        '''
        Predictions
        '''
        X_tfh = tensorize_segments(X, self.n_h, n_fut=self.n_fut)
        n_t, n_f, n_hall = X_tfh.shape
        x = torch.from_numpy(X_tfh.reshape(n_t, n_f*n_hall)).type(torch.FloatTensor)

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
        Show the "kernel"
        '''
        show_strf(self.info['k_fh'])

    def dump(self):
        return self.info

    def reload(self, info):
        '''
        Reinitialise network from result of self.dump()
        '''
        self.n_f, self.n_h, self.n_fut = info['n_f'], info['n_h'], info['n_fut']
        self.linear = torch.nn.Linear(self.n_f*(self.n_h+self.n_fut), 1, bias=True)
        self.linear.bias = torch.nn.Parameter(torch.from_numpy(info['b']))
        self.linear.weight = torch.nn.Parameter(torch.from_numpy(info['w']))
        if torch.cuda.is_available():
            self.linear = self.linear.cuda()

class TorchNRF(torch.nn.Module):
    def __init__(self, n_hidden=3, n_h=15, n_fut=0,
                 learning_rate=1e-4, epochs=15000, lamb=1e-5):
        super(TorchNRF, self).__init__()
        self.n_f = None
        self.n_h = n_h
        self.n_fut = n_fut
        self.linear1 = None
        self.linear2 = None
        self.linear3 = None
        self.n_hidden = n_hidden
        self.sigmoid = torch.nn.Sigmoid()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lamb = lamb
        self.x_prediction = None
        self.y_prediction = None
        self.info = None

        self.criterion = torch.nn.MSELoss()
        self.optimizer = None

    def fit(self, X=None, y=None, plot_loss=False, early_stop=False):
        X_tfh = tensorize_segments(X, self.n_h, n_fut=self.n_fut)
        n_t, self.n_f, n_hall = X_tfh.shape
        y = concatenate_segments(y, mean=True)

        n_train = int(0.9 * n_t)

        x_train = Variable(torch.from_numpy(X_tfh[:n_train, :].reshape(-1, self.n_f*n_hall)).type(torch.FloatTensor))
        y_train = Variable(torch.from_numpy(y[:n_train].reshape(-1, 1)).type(torch.FloatTensor))

        x_tune = Variable(torch.from_numpy(X_tfh[n_train:, :].reshape(-1, self.n_f*n_hall)).type(torch.FloatTensor))
        y_tune = Variable(torch.from_numpy(y[n_train:].reshape(-1, 1)).type(torch.FloatTensor))

        self.linear1 = torch.nn.Linear(self.n_f*n_hall, self.n_hidden, bias=True)
        self.linear2 = torch.nn.Linear(self.n_hidden, 1, bias=True)
        self.linear3 = torch.nn.Linear(1, 1, bias=True)

        if torch.cuda.is_available():
            x_train = x_train.cuda()
            y_train = y_train.cuda()
            x_tune = x_tune.cuda()
            y_tune = y_tune.cuda()
            self.linear1 = self.linear1.cuda()
            self.linear2 = self.linear2.cuda()
            self.linear3 = self.linear3.cuda()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        saved_training_loss = np.zeros(self.epochs)
        saved_tuning_loss = np.zeros(self.epochs)

        best_tuning_loss = np.Inf
        best_network = None
        best_epoch = -1
        epochs_since_best_tuning_loss = 0

        for epoch in range(self.epochs):

            # clear gradient buffers
            self.optimizer.zero_grad()
            outputs = self.forward(x_train)
            loss = self.criterion(outputs, y_train) + \
                self.lamb*(self.linear1.weight.norm(p=1))

            loss.backward()
            self.optimizer.step()

            outputs = self.forward(x_tune)
            tuning_loss = self.criterion(outputs, y_tune) + \
                self.lamb*(self.linear1.weight.norm(p=1)) # unregularized

            if tuning_loss.item() < best_tuning_loss:
                best_epoch = epoch
                best_tuning_loss = tuning_loss.item()
                epochs_since_best_tuning_loss = 0
                best_network = self.get_params()

            else:
                epochs_since_best_tuning_loss = epochs_since_best_tuning_loss + 1

            saved_training_loss[epoch] = loss.item()
            saved_tuning_loss[epoch] = tuning_loss.item()
            #print('epoch {}, loss {}'.format(epoch, loss.item()))

            # if early_stop and epoch > (self.epochs/3) and epochs_since_best_tuning_loss > 500:
            #     break

        saved_training_loss = saved_training_loss[:epoch+1]
        saved_tuning_loss = saved_tuning_loss[:epoch+1]

        if early_stop:
            self.reload(best_network)
        else:
            best_epoch = epoch

        if plot_loss:
            plt.plot(np.arange(len(saved_training_loss)), saved_training_loss)
            plt.plot(np.arange(len(saved_tuning_loss)), saved_tuning_loss)
            plt.plot((best_epoch, best_epoch), plt.ylim())
            plt.legend(('Training', 'Tuning'))

        self.info = best_network

    def get_params(self):
        w1 = self.linear1.weight.detach().cpu().numpy()
        return {type: 'TorchNRF',
                'n_f': self.n_f,
                'n_h': self.n_h,
                'n_fut': self.n_fut,
                'n_hidden': self.n_hidden,
                'k_fh': w1.reshape(self.n_hidden, self.n_f, (self.n_h+self.n_fut)).transpose((1, 0, 2)). \
                   reshape(self.n_f, (self.n_h+self.n_fut)*self.n_hidden),
                'b1': self.linear1.bias.detach().cpu().numpy(),
                'w1': w1,
                'b2': self.linear2.bias.detach().cpu().numpy(),
                'w2': self.linear2.weight.detach().cpu().numpy(),
                'b3': self.linear3.bias.detach().cpu().numpy(),
                'w3': self.linear3.weight.detach().cpu().numpy(),
               }

    def forward(self, x):
        out = self.linear3(self.sigmoid(self.linear2(self.sigmoid(self.linear1(x)))))
        return out

    def predict(self, X=None):
        '''
        Predictions
        '''
        with torch.no_grad():
            X_tfh = tensorize_segments(X, self.n_h, n_fut=self.n_fut)
            n_t, n_f, n_hall = X_tfh.shape
            x = torch.from_numpy(X_tfh.reshape(n_t, n_f*n_hall)).type(torch.FloatTensor)
            if torch.cuda.is_available():
                x = x.cuda()
            out = self.forward(x)
        return out.detach().cpu().numpy().ravel()

    def store_prediction(self, X=None, y=None):
        '''
        Tensorize and store prediction set for later use
        '''
        X_tfh = tensorize_segments(X, self.n_h, n_fut=self.n_fut)
        n_t, n_f, n_hall = X_tfh.shape
        self.x_prediction = torch.from_numpy(X_tfh.reshape(n_t, n_f*n_hall)).type(torch.FloatTensor)
        self.y_prediction = concatenate_segments(y)

    def predict_stored(self):
        '''
        Predictions on stored set
        '''
        if self.x_prediction is None:
            raise ValueError('self.x_prediction is not set')
        x = self.x_prediction
        with torch.no_grad():
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

    def score_stored(self, sample_weight=None):
        '''
        Score stored data
        '''
        y_hat = self.predict_stored()

        if self.y_prediction is None:
            raise ValueError('self.y_prediction is not set')
        y = self.y_prediction

        if len(y.shape) == 1:
            return np.corrcoef(y, y_hat)[0, 1]

        return calc_CC_norm(y, y_hat)

    def show(self):
        '''
        Show the "kernel"
        '''

        show_strf(self.info['k_fh'])

    def dump(self):
        '''
        Dump most important info in pickleable format
        '''
        return self.info

    def reload(self, info):
        '''
        Reinitialise network from result of self.dump()
        '''
        self.n_f, self.n_h, self.n_fut = info['n_f'], info['n_h'], info['n_fut']
        self.n_hidden = info['n_hidden']
        self.linear1 = torch.nn.Linear(self.n_f*(self.n_h+self.n_fut), self.n_hidden, bias=True)
        self.linear1.bias = torch.nn.Parameter(torch.from_numpy(info['b1']))
        self.linear1.weight = torch.nn.Parameter(torch.from_numpy(info['w1']))
        self.linear2 = torch.nn.Linear(self.n_hidden, 1, bias=True)
        self.linear2.bias = torch.nn.Parameter(torch.from_numpy(info['b2']))
        self.linear2.weight = torch.nn.Parameter(torch.from_numpy(info['w2']))
        self.linear3 = torch.nn.Linear(1, 1, bias=True)
        self.linear3.bias = torch.nn.Parameter(torch.from_numpy(info['b3']))
        self.linear3.weight = torch.nn.Parameter(torch.from_numpy(info['w3']))

        if torch.cuda.is_available():
            self.linear1 = self.linear1.cuda()
            self.linear2 = self.linear2.cuda()
            self.linear3 = self.linear3.cuda()
