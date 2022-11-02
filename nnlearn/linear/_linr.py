import numpy as np
from rich.progress import track

from nnlearn.reporting import GdReport
from nnlearn.nanograd import Var
from nnlearn.base import GdBase

class LinearRegression(GdBase, GdReport):

    def __init__(self,
        optimizer='gd_backp',
        loss_func='mse',
        epochs=50,
        batch_size=1.0,
        shuffle=False,
        lr=.01,
        bias=True,
        figpath=""):
        
        # Common attributes to models optimized via GD
        GdBase.__init__(self,
                        batch_size,
                        shuffle,
                        loss_func,
                        epochs,
                        lr)

        # Reporting
        GdReport.__init__(self, figpath, 'reg')

        # LR specific
        self.optimizer = optimizer
        self.bias = bias
        self._theta = None

    def _zero_grads(self):
        for w in self._theta[:, 0]:
            w.grad = 0

    def _update_weights(self):
        for w in self._theta[:, 0]:
            w.v -= self.lr * w.grad

    def _forward(self, X):
        return (X @ self._theta)[:, 0]

    def _train(self):
        for epoch in track(range(1, self.epochs + 1), "Training..."):
            self._reshuffle()
            X_batches, y_batches = self._get_batches()

            batch = 1
            losses = []
            for X, y in zip(X_batches, y_batches):

                # Predict
                yhat = self._forward(X)

                # Compute loss based on the prediction
                loss = self.loss_func(y, yhat)
                losses.append(loss.v)

                # reset gradients of variables to zero
                self._zero_grads() 

                # backward propagate
                loss.backward()

                # update weights
                self._update_weights()
                
                # Increase batch number
                batch += 1
            
            # Epoch evaluation
            yhat_train = self._arr_to_val(self._forward(self.Xv))
            y_train = self._arr_to_val(self.yv)
            self.eval_epoch(epoch, losses, y_train, yhat_train)

        self.create_report(self.loss_func_name, self.batch_size, self.lr)

    def _initialize_parameters(self, m):
        m = m + 1 if self.bias else m
        self._theta = np.random.normal(0, 1, m).reshape(-1, 1)
        if self.optimizer == 'gd_backp':
            self._theta = self._arr_to_var(self._theta)

    def _add_constant_column(self, X):
        if self.bias: 
            X = np.hstack((X, np.ones((X.shape[0], 1), dtype=X.dtype)))
        return X
 
    def fit(self, X, y):
        self._initialize_parameters(X.shape[1])
        X = self._add_constant_column(X)
        self._preprocessing(X, y)
        self._train()
    
    def predict(self, X):
        X = self._add_constant_column(X)
        Xv = self._arr_to_var(X)
        yhat = self._arr_to_val(self._forward(Xv))
        return yhat
