from collections import Counter
from nnlearn.datasets import load_iris
from nnlearn.metrics import accuracy_score
from nnlearn.tree import Node
from nnlearn.tree import DecisionTree
import numpy as np
from sklearn.model_selection import train_test_split


class DecisionTreeClassifier:

    """
    The `DecisionTreeClassifier` is a tree based ML model used for classification.

    Parameters
    ----------
    criterion_name : str, optional
        Name of the metric based on which to define purity of tree nodes.
    splitter : str, optional
        Name of the method to use for splitting nodes.
    max_depth : int, optional
        Maximum depth of the tree.
    min_samples_split : int, optional
        Minimum number of samples present within given node in order for it to become an internal node.
    min_samples_leaf : int, optional
        Mimimum number of leaves to be present within a leaf.
    max_features : int, optional
        Maximum number of features to take into account when deciding on how to split the node.
    random_state : int, optional
        When you are not using all features to split the node and only selecting randomly a subset, then this will ensure reproducibility.
    max_leaf_nodes : int, optional
        What is the maximum of leaf nodes given tree can have.
    min_impurity_decrease : float, optional
        What is the minimum amount of impurity decrease for which is worth to split the given node.

    Todo
    ----

    - Implement min_impurity_decrease.

    - Implement min_samples_split.

    """

    def __init__(self,
                criterion_name='gini',
                splitter="best",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features=None,
                random_state=42,
                max_leaf_nodes=None,
                min_impurity_decrease=0):
        
        self.tree = DecisionTree(
                criterion_name=criterion_name,
                splitter=splitter,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=random_state,
                max_leaf_nodes=max_leaf_nodes,
                min_impurity_decrease=min_impurity_decrease
        )

        self.classes = None
        self.n_features_in = None
        self.n_outputs = None
    
    def fit(self, X, y):

        """
        Train the model.

        Parameters
        ----------
        X : 2d array
            Training data.
        y : 1d array
            Ground truth values.
        """
        
        self.classes = np.unique(y)
        self.n_features_in = X.shape[1]
        self.n_outputs = X.shape[1]

        self.tree.root = Node(X, y, self.tree)
        self.tree.root.split()
    
    def predict(self, X):

        """
        Predicts labels for given records.

        Parameters
        ----------
        X : 2d array
            Data based on which to predict labels.
        
        Returns
        -------
        1d array
            Predicted labels.
        """

        result = []
        for x in X:
            result.append(self.tree._get_prediction(x, self.tree.root))

        return np.array(result)
