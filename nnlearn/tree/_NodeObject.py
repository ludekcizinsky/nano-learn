class Node:

    """
    Node object serves as a core element as part of the deciosion tree data structure.

    Parameters
    ----------
    X : 2d array
        Records which the given nodes holds.
    y : 1d array
        Labels for the records.
    tree : :class:`DecisionTree`
        Decision tree object.
    impurity: float, optional
        Impurity of this node.
    
    Attributes
    ----------
    left : :class:`Node`
        Left child.
    right : :class:`Node`
        Right child.
    threshold : float
        Value where to make split.
    feature : int
        Index of feature within ``X`` based on which to do the split.
    """

    def __init__(self, X, y, tree, impurity=None, **kwargs):
        self.X = X
        self.y = y
        self.n = len(self.y)
        self.tree = tree
        self.impurity = impurity
        self.left = None
        self.right = None
        self.threshold = None
        self.feature = None

    def split(self):

        if not self.is_leaf_node():

            # Save the results as follows: [feature_id, (metadata)]
            results = []

            for j in range(self.X.shape[1]):

                # Save computed impurities and boundaries
                impurities = []

                # Extract the feature values
                feature = sorted(list(set(self.X[:, j])))

                # Iterate over possible splits
                for i in range(1, len(feature)):

                    # Compute decision boundary
                    dcb = (feature[i] + feature[i - 1])/2

                    # Split the labels accordingly, and compute gini for both nodes
                    left = self.y[self.X[:, j] < dcb]
                    left_imp = self.tree.criterion(left)
                    right = self.y[self.X[:, j] > dcb]
                    right_imp = self.tree.criterion(right)
                    if len(left) == 0 or len(right) == 0:
                        pass

                    # Combine the results using weighted average, and save it
                    total = (len(left)/self.n)*left_imp + (len(right)/self.n)*right_imp
                    impurities.append((dcb, total, left_imp, right_imp))

                # Sort the impurities and choose the boundary with highest impurity
                imp_sorted = sorted(impurities, key=lambda x: x[1], reverse=False)
                results.append([j] + list(imp_sorted[0]))

            # Get the best feature based on impurity
            best_res = sorted(results, key=lambda x: x[2], reverse=False)[0]

            # Assign threshold, which feature
            self.threshold = best_res[1]
            self.feature = best_res[0]

            # Make the split (if it makes sense)
            left_imp, right_imp = best_res[3], best_res[4]

            # * Left
            if (self.impurity is None) or (self.impurity is not None and left_imp < self.impurity):

                # ** Prepare the data
                left_mask = self.X[:, self.feature] < self.threshold
                left_X, left_y = self.X[left_mask], self.y[left_mask]

                # ** Add left child
                self.left = Node(X=left_X,
                                 y=left_y,
                                 tree=self.tree,
                                 impurity=left_imp)
                self.left.split()

            # * Right
            if (self.impurity is None) or (self.impurity is not None and right_imp < self.impurity):

                # ** Prepare the data
                right_mask = self.X[:, self.feature] > self.threshold
                right_X, right_y = self.X[right_mask], self.y[right_mask]

                # ** Add right child
                self.right = Node(X=right_X,
                                  y=right_y,
                                  tree=self.tree,
                                  impurity=right_imp)
                self.right.split()

    def is_leaf_node(self):
        if len(self.y) == 1:
            return True
        elif self.impurity is not None and self.impurity < 0.001:
            return True
        else:
            return False
