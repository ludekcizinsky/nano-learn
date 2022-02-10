from nnlearn.metrics import gini_score, entropy_score
from nnlearn.exceptions import CriterionFunctionNotFound
from collections import Counter

class DecisionTree:

    """
    Decision Tree data structure.

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
    
    Notes
    -----
    This implementation uses node objects as an underlying data structure. Each node has
    left and right child if it is an internal node or root.
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

        self.criterion_name = criterion_name
        self.splitter = splitter
        self.root = None
        self.criterion = None
        self._run_setup()

    def _get_prediction(self, x, node):

        if not node.is_leaf_node():

            if x[node.feature] > node.threshold and node.right:
                return self._get_prediction(x, node.right)
            elif x[node.feature] < node.threshold and node.left:
                return self._get_prediction(x, node.left)

        return sorted([(key, value) for key, value in Counter(node.y).items()], reverse=True, key=lambda x: x[1])[0][0]
    
    def _run_setup(self):

        if self.criterion_name == "gini":
                self.criterion = gini_score
        elif self.criterion_name == "entropy":
            self.criterion = entropy_score
        else:
            raise CriterionFunctionNotFound(f'Specified function {self.criterion_name} is NOT implemented. Please specify a function which is implemented.')
