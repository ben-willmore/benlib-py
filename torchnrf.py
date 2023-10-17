'''
PyTorch model estimation and display functions
'''

# pylint: disable=C0103, R0912, R0914

import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.autograd import Variable
from benlib.utils import calc_CC_norm
from benlib.strf import show_strf

try:
    if torch.cuda.is_available():
        print('CUDA available')
    elif torch.backends.mps.is_available():
        print('MPS available')
except:
    pass

class TorchLinearRegression(torch.nn.Module):
    def __init__(self, learning_rate=1e-2, epochs=2500, lamb=1e-2):
        super(TorchLinearRegression, self).__init__()
        self.linear = None
        self.sigmoid = torch.nn.Sigmoid()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lamb = lamb

        self.criterion = torch.nn.MSELoss()
        self.optimizer = None

        self.info = None

    def fit(self, X=None, y=None, plot_loss=False):
        n_t, n_f, n_h = X.shape
        x_train = Variable(torch.from_numpy(X.reshape(n_t, n_f*n_h)).type(torch.FloatTensor))
        y_train = Variable(torch.from_numpy(y.reshape(n_t, 1)).type(torch.FloatTensor))

        self.linear = torch.nn.Linear(n_f*n_h, 1, bias=True)
        if torch.cuda.is_available():
            x_train = x_train.cuda()
            y_train = y_train.cuda()
            self.linear = self.linear.cuda()
        elif torch.backends.mps.is_available():
            mps_device = torch.device('mps')
            x_train = x_train.to(mps_device)
            y_train = y_train.to(mps_device)
            self.linear = self.linear.to(mps_device)

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
                     'n_f': n_f,
                     'n_h': n_h,
                     'c': self.linear.bias.detach().cpu().numpy(),
                     'k_fh': w.reshape(n_f, n_h), # for convenience,
                     'b': self.linear.bias.detach().cpu().numpy(),
                     'w': w
                     }

    def forward(self, x):
        return self.linear(x)

    def predict(self, X=None):
        '''
        Predictions
        '''
        n_t, n_f, n_h = X.shape
        x = torch.from_numpy(X.reshape(n_t, n_f*n_h)).type(torch.FloatTensor)

        if torch.cuda.is_available():
            x = x.cuda()
        elif torch.backends.mps.is_available():
            mps_device = torch.device('mps')
            x = x.to(mps_device)
        out = self.forward(x)
        return out.detach().cpu().numpy().ravel()

    def score(self, X=None, y=None, sample_weight=None):
        '''
        Score
        '''
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
        self.info = info
        self.linear = torch.nn.Linear(info['n_f']*info['n_h'], 1, bias=True)
        self.linear.bias = torch.nn.Parameter(torch.from_numpy(info['b']))
        self.linear.weight = torch.nn.Parameter(torch.from_numpy(info['w']))
        if torch.cuda.is_available():
            self.linear = self.linear.cuda()
        elif torch.backends.mps.is_available():
            mps_device = torch.device('mps')
            self.linear = self.linear.to(mps_device)

class TorchNRF(torch.nn.Module):
    def __init__(self, n_hidden=3,
                 learning_rate=1e-4, epochs=15000, lamb=1e-5):
        super(TorchNRF, self).__init__()
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
        n_t, n_f, n_h = X.shape

        n_train = int(0.9 * n_t)

        x_train = Variable(torch.from_numpy(X[:n_train, :].reshape(-1, n_f*n_h)).type(torch.FloatTensor))
        y_train = Variable(torch.from_numpy(y[:n_train].reshape(-1, 1)).type(torch.FloatTensor))

        x_tune = Variable(torch.from_numpy(X[n_train:, :].reshape(-1, n_f*n_h)).type(torch.FloatTensor))
        y_tune = Variable(torch.from_numpy(y[n_train:].reshape(-1, 1)).type(torch.FloatTensor))

        self.linear1 = torch.nn.Linear(n_f*n_h, self.n_hidden, bias=True)
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
        elif torch.backends.mps.is_available():
            mps_device = torch.device('mps')
            x_train = x_train.to(mps_device)
            y_train = y_train.to(mps_device)
            x_tune = x_tune.to(mps_device)
            y_tune = y_tune.to(mps_device)
            self.linear1 = self.linear1.to(mps_device)
            self.linear2 = self.linear2.to(mps_device)
            self.linear3 = self.linear3.to(mps_device)

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
                best_network = self.get_params(n_f, n_h)

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

    def get_params(self, n_f, n_h):
        w1 = self.linear1.weight.detach().cpu().numpy()
        return {type: 'TorchNRF',
                'n_f': n_f,
                'n_h': n_h,
                'n_hidden': self.n_hidden,
                'lamb': self.lamb,
                'k_fh': w1.reshape(self.n_hidden, n_f, n_h).transpose((1, 0, 2)). \
                   reshape(n_f, n_h*self.n_hidden),
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
            n_t, n_f, n_h = X.shape
            x = torch.from_numpy(X.reshape(n_t, n_f*n_h)).type(torch.FloatTensor)
            if torch.cuda.is_available():
                x = x.cuda()
            elif torch.backends.mps.is_available():
                mps_device = torch.device('mps')
                x = x.to(mps_device)
            out = self.forward(x)
        return out.detach().cpu().numpy().ravel()

    def store_prediction(self, X=None, y=None):
        '''
        Tensorize and store prediction set for later use
        '''
        n_t, n_f, n_h = X.shape
        self.x_prediction = torch.from_numpy(X.reshape(n_t, n_f*n_h)).type(torch.FloatTensor)
        self.y_prediction = y

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
            elif torch.backends.mps.is_available():
                mps_device = torch.device('mps')
                x = x.to(mps_device)
            out = self.forward(x)
        return out.detach().cpu().numpy().ravel()

    def score(self, X=None, y=None, sample_weight=None):
        '''
        Score
        '''
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
        self.info = info
        self.n_hidden = info['n_hidden']
        self.linear1 = torch.nn.Linear(info['n_f']*info['n_h'], self.n_hidden, bias=True)
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
        elif torch.backends.mps.is_available():
            mps_device = torch.device('mps')
            self.linear1 = self.linear1.to(mps_device)
            self.linear2 = self.linear2.to(mps_device)
            self.linear3 = self.linear3.to(mps_device)

class TorchNRFCV():
    '''
    Cross-validated hyperparameter search for TorchNRF
    '''

    def __init__(self):
        self.best_score = None
        self.best_model = None
        self.best_lambda = None
        self.best_n_hidden = None

    def fit(self, X=None, y=None, lambdas=None, n_hidden_range=None):
        if lambdas is None:
            lambdas = 10.0**np.arange(-6, 0, 1)
        if n_hidden_range is None:
            n_hidden_range = [1,16,32]

        n_t = len(y)
        n_cv = n_t//10
        train_idx = np.arange(0, n_t-n_cv)
        test_idx = np.arange(n_t-n_cv, n_t)

        self.best_score = -100000
        self.best_model = None
        for lamb in lambdas:
            for n_hidden in n_hidden_range:
                model = TorchNRF(n_hidden=n_hidden, lamb=lamb)
                model.fit(X[train_idx,:,:], y[train_idx])
                score = model.score(X[test_idx,:,:], y[test_idx])
                print(lamb, n_hidden, score)
                if score > self.best_score:
                    self.best_model = model
                    self.best_score = score
                    self.best_lambda = lamb
                    self.best_n_hidden = n_hidden

    def predict(self, X=None):
        return self.best_model.predict(X=X)

    def score(self, X=None, y=None, sample_weight=None):
        return self.best_model.score(X=X, y=y, sample_weight=sample_weight)

    def dump(self):
        dump = self.best_model.dump()
        dump['best_score'] = self.best_score
        dump['best_lambda'] = self.best_lambda
        dump['best_n_hidden'] = self.best_n_hidden
        return dump
