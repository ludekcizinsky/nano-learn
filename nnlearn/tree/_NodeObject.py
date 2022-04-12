from random import choice

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
        self.is_leaf = False

    def split(self):
        """Split the node if it is possible.
        """
        
        # Compute the optimal split along with node's impurity
        self._get_optimal_split()
        
        # Split the node if it is possible
        if self._is_splittable(): 
            self._make_optimal_split()
            self.right.split()
            self.left.split()
        else:
            self.is_leaf = True
    
    def is_leaf_node(self):
        """Return if the current node is a leaf node.
        """
        return self.is_leaf 

    def _make_optimal_split(self):
        """Makes the split according to the optimal criteria.
        """

        # Left
        left_mask = self.X[:, self.feature] < self.threshold
        left_X, left_y = self.X[left_mask], self.y[left_mask]

        self.left = Node(X=left_X,
                         y=left_y,
                         tree=self.tree)
        
        # Right
        right_mask = self.X[:, self.feature] > self.threshold
        right_X, right_y = self.X[right_mask], self.y[right_mask]

        self.right = Node(X=right_X,
                          y=right_y,
                          tree=self.tree)

    def _get_optimal_split(self):
        """Returns the most optimal split.

        Most optimal split is defined in terms
        which feature to use and what threshold.
        """

        # Save the results as follows: [feature_id, (metadata)]
        results = []
        
        # Decide on which features to consider
        m = self.X.shape[1]
        all_j = [v for v in range(m)]
        if self.tree.max_features is None:
          J = all_j
        elif self.tree.max_features != m:
          J = [choice(all_j) for _ in range(self.tree.max_features)]
        else:
          raise ValueError("Unexpected error. Please report this as an issue.")

        # Iterate over all features
        for j in J:

            # Save computed impurities and boundaries
            impurities = []

            # Extract the feature values
            feature = sorted(list(set(self.X[:, j])))

            if len(feature) == 1:
              continue

            # Iterate over possible splits
            for i in range(1, len(feature)):

                # Compute decision boundary
                dcb = (feature[i] + feature[i - 1])/2

                # Split the labels accordingly, and compute impurity for both nodes
                left = self.y[self.X[:, j] < dcb]
                left_imp = self.tree.criterion(left)
                right = self.y[self.X[:, j] > dcb]
                right_imp = self.tree.criterion(right)
                if len(left) == 0 or len(right) == 0:
                    raise ValuerError("Unexpected error. Report this as an issue please.")

                # Combine the results using weighted average, and save it
                total = (len(left)/self.n)*left_imp + (len(right)/self.n)*right_imp
                impurities.append((dcb, total, left_imp, right_imp))

            # Sort the impurities and choose the boundary with highest impurity
            imp_sorted = sorted(impurities, key=lambda x: x[1], reverse=False)
            results.append([j] + list(imp_sorted[0]))
        
        # Get the best feature based on impurity
        if results:
            best_res = sorted(results, key=lambda x: x[2], reverse=False)[0]
            
            # Update the model attributes
            self.feature, self.threshold, self.impurity = best_res[0], best_res[1], best_res[2]

    def _is_splittable(self):
        """Returns if the given node can be further splitted.

        Returns
        -------
        res : bool
        """
        
        # At least 2 samples (default) or any other number
        # defined by client
        c1 = len(self.y) >= self.tree.min_samples_split

        # Still impure
        c2 = self.impurity is not None and self.impurity > 0.001


        res = c1 and c2

        return res

