import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

from rich.table import Table
from rich.progress import track
from rich.columns import Columns
from rich.panel import Panel
from rich.markdown import Markdown

from nnlearn.metrics import mean_cross_entropy_score, mean_squared_error, accuracy_score
from nnlearn.nanograd import Var
from nnlearn.base import GdBase

class FFNN(GdBase):

    """
    Feed forward neural network is a machine learning model which is used for all
    kinds of tasks from regression to classification.

    This class serves as a high level API to setup and train neural network.

    Parameters
    -----------
    layers : iterable
        One dimensional iterable where each item is a :class:`DenseLayer`.
    loss_func : str, optional
        Loss function to use for training.
    epochs : int, optional
        How many times do you want to feed the whole dataset trough the network.
    batch_size : float or int, optional
        Within each epoch, you have the option to feed the network in batches. Here you
        can specify its size.
    shuffle : bool, optional
        Do you want to shuffle the data after each epoch. (Shuffle rows)
    optimizer : str, optional
        Name of the optimizer to tune network's parameters.

    Notes
    -----

    ``Layers``
    Esentially, there are three types of layers - `input, hidden, output.`
    Each layer contains certain number of neurons. The number neurons in the
    input layer is determined based on the number of features in the provided
    dataset. Similarly, in case of output layer, it depends on the task at hand.
    For example, if you are doing regression, then you only need one neuron
    in the output layer or if you are doing multiclass classification, then
    you need same number of neurons as there is distinct classes in the dataset.
    """

    def __init__(self,
        layers,
        logger,
        loss_func='mse',
        epochs=50,
        batch_size=1.0,
        shuffle=False,
        lr=.01):

        GdBase.__init__(self, batch_size, shuffle)

        self.layers = layers
        self.loss_func_name = loss_func
        self.epochs = epochs
        self.lr = lr
        
        self.loss_func = None
        self.n = None
        self.m = None

        self.table = None
        self.fig = None
        self.report = None

        self._run_setup()
    
    def _run_setup(self):

        """
        The goal of this function is to run all neccessary operations
        after client initialies this class.
        """
        
        # Loss functions
        if self.loss_func_name == 'mse':
            self.loss_func = mean_squared_error
        elif self.loss_func_name == 'cross_entropy':
            self.loss_func = mean_cross_entropy_score
 
    def _forward(self, X):

        res = X
        for l in self.layers:
            res = l.step(res)
        return res
    
    def _zero_grads(self):

        for l in self.layers:
            l._zero_grads()

    def _update_weights(self):

        for l in self.layers:
            l._update_weights(self.lr)

    def _eval_epoch(self, losses):

        # Mean loss
        mean_loss = sum(losses)/len(losses)
        
        # Accuracy
        yhat = np.argmax(self.predict_proba(self.Xv), axis=1)
        y = self._arr_to_val(self.yv)
        acc = accuracy_score(y, yhat)

        return mean_loss, acc

    def _setup_report_table(self):
        
        self.table = table = Table()
        self.table.add_column("Epoch", justify="center", style="subtle")
        self.table.add_column("Loss", justify="center", style="rose")
        self.table.add_column("Accuracy", justify="center", style="love")

    def _create_report(self, losses, accuracies):
        
        # Define plot
        self.fig, axs = plt.subplots(2, 1, sharex=True)
        plt.tight_layout();
        self.fig.subplots_adjust(wspace=0.2, hspace=.2);

        # Define epochs
        epochs = np.arange(1, self.epochs + 1, 1)

        # Epochs vs loss
        sns.lineplot(x=epochs, y=losses, ax=axs[0], label='loss')
        axs[0].set_title("Epoch # vs loss")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        
        # Epochs vs accuracy
        sns.lineplot(x=epochs, y=accuracies, ax=axs[1], label='accuracy') 
        axs[1].set_title("Epoch # vs accuracy")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Accuracy")
        
        # Create markdown
        md = ""
        md += "## Hyper-parameters\n"
        md += "Following hyper-parameters have been used:\n"
        md += f"- Epochs: {self.epochs}\n"
        md += f"- Loss func: {self.loss_func_name}\n"
        md += f"- Batch size: {self.batch_size}\n"
        md += f"- LR: {self.lr}\n"
        md += "## Training plot\n"
        md += "📈 See training plot [here](training.png)\n"

        panel_1 = Panel.fit(self.table, title="table of training", width=35)
        panel_2 = Panel.fit(Markdown(md), title="training information", width=35)
        self.report = Columns([panel_1, panel_2]) 

    def _train(self):
        
        self._setup_report_table()

        mean_losses = []
        accuracies = []
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
            
            # Compute metrics for given epoch
            mean_loss, acc = self._eval_epoch(losses)

            # Save them for plotting
            mean_losses.append(mean_loss)
            accuracies.append(acc)

            # Create table row
            epoch = "{:05d}".format(epoch)
            mean_loss = "{:06.3f}".format(mean_loss)
            acc = "{:04.2f}".format(acc)
            self.table.add_row(epoch, mean_loss, acc)

        self._create_report(mean_losses, accuracies)
        print()
          
 
    def fit(self, X, y):
        """Find the optimal parameters.

        Parameters
        ----------
        X : 2d array 
          Training data.
        y : 1d array
          Training labels.
        """

        self.n, self.m = X.shape
        self._preprocessing(X, y)
        self._train()

    def predict_proba(self, X):

        return  self._arr_to_val(self._forward(X))

    def predict(self, X):

        Xv = self._arr_to_var(X)
        probs = self.predict_proba(Xv)
        return np.argmax(probs, axis=1)

