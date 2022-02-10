from _Nanograd import Var, _arr_to_var
import numpy as np


class NeuralNetwork:

    """
    Neural network is a machine learning model which is used for all
    kinds of tasks from regression to classification.

    This class serves as a high level API to setup and train neural network.

    Parameters
    -----------
    layers : iterable
        One dimensional iterable where each item is a :class:`Layer`.
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

    ``Loss functions``
    To be added.

    ``Epoch, batch size``
    To be added.

    ``Training process``
    To be added.

    Todo
    ----

    - Complete notes section

    - Make preprocessing section more robust

    """

    def __init__(self, layers, loss_func='mse', epochs=50, batch_size=1.0, shuffle=False, optimizer='gd'):

        # User inputed attributes
        self.layers = layers
        self.loss_func = loss_func
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.optimizer = optimizer

        # Derived attributes
        # * Training dataset
        self.X = None # Original
        self.y = None # - || -
        self.Xv = None # Uses Var object instead of numbers
        self.yv = None # - || -
        self.n = None # Number of training records
        self.m = None # Number of features in the training dataset
    
    def _initial_setup(self):

        """
        The goal of this function is to run all neccessary operations
        after client initialies this class.
        """

        pass
    
    def _preprocessing(self):

        """ 
        Preprocesses provided training data.

        Notes
        -----
        Following methods are applied:
        - Transform training data into Var object
        - Transform batch size from proportion to actual size if neccessary
        """

        self.Xv, self.yv = _arr_to_var(X), _arr_to_var(y)

        if isinstance(self.batch_size, float):
            self.batch_size = int(self.n*self,batch_size)
    
    def _get_batches(self):

        """
        Splits the given training dataset into the
        batches of given size.

        Returns
        -------
        X_batches : list
            List with batches where each batch is just a subsample of X.
        y_batches : list
            List with batches where each batch is just a subsample of y.
        """
        
        choices = set([i for i in range(self.n)])
        X_batches = []
        y_batches = []
        while len(choices) > 0:

            size = min(self.batch_size, len(choices))
            selected = np.random.choice(np.array(choices), size=size, replace=False)

            X_batches.append(self.Xv[selected])
            y_batches.append(self.yv[selected])

            choices = choices - set(selected)
        
        return X_batches, y_batches
    
    def _reshuffle(self):

        """
        Reshuffles the order of rows in the training dataset.
        """

        if self.shuffle:

            rows_index = np.array([i for i in range(self.n)])
            np.random.shuffle(rows_index) # In-place

            self.Xv = self.Xv[rows_index]
            self.yv = self.yv[rows_index]
    
    def _train(self):

        for epoch in range(1, self.epochs + 1):

            X_batches, y_batches = self._get_batches()

            for X, y in zip(X_batches, y_batches):
                pass
            
            self._reshuffle()
            

        
    
    def fit(self, X, y):
        
        # Update classes attributes
        self.X, self.y = X, y

        # Run preprocessing pipeline
        self.preprocessing()

        # Run training pipeline

        

    
    def predict(self, X, y):
        pass
