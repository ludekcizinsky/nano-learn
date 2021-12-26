from collections import Counter
from nnlearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import numpy as np


class Node:

    def __init__(self, **kwargs) -> None:
        self.X = kwargs.get("X")
        self.y = kwargs.get("y")
        self.impurity = kwargs.get("impurity")
        self.n = len(self.y)
        self.criterion_name = kwargs.get("criterion")
        self.criterion = None
        self.left = None
        self.right = None
        self.threshold = None
        self.feature = None

        # Set up criterion
        if self.criterion_name:
            if self.criterion_name == "gini":
                self.criterion = self.gini
        else:
            self.criterion = self.gini

    def split(self):

        if not self.is_leaf_node():

            print("Splitting node...\n")

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
                    left_imp = self.criterion(left)
                    right = self.y[self.X[:, j] > dcb]
                    right_imp = self.criterion(right)
                    if len(left) == 0 or len(right) == 0:
                        pass

                    # Combine the results using weighted average, and save it
                    total = (len(left)/self.n)*left_imp + \
                        (len(right)/self.n)*right_imp
                    impurities.append((dcb, total, left_imp, right_imp))

                # Sort the impurities and choose the boundary with highest impurity
                imp_sorted = sorted(
                    impurities, key=lambda x: x[1], reverse=False)
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

                print("Creating left node")

                # ** Prepare the data
                left_mask = self.X[:, self.feature] < self.threshold
                left_X, left_y = self.X[left_mask], self.y[left_mask]

                print(left_imp)
                print(left_y)
                print()
                # ** Add left child
                self.left = Node(X=left_X,
                                 y=left_y,
                                 criterion=self.criterion_name,
                                 impurity=left_imp)
                self.left.split()

            # * Right
            if (self.impurity is None) or (self.impurity is not None and right_imp < self.impurity):

                print("Creating right node")

                # ** Prepare the data
                right_mask = self.X[:, self.feature] > self.threshold
                right_X, right_y = self.X[right_mask], self.y[right_mask]

                print(right_imp)
                print(right_y)
                print()
                # ** Add right child
                self.right = Node(X=right_X,
                                  y=right_y,
                                  criterion=self.criterion_name,
                                  impurity=right_imp)
                self.right.split()

    def gini(self, y):

        counts = Counter(y)
        n = len(y)
        result = 0
        for count in counts.values():
            result += (count/n)*(1 - count/n)

        return result

    def is_leaf_node(self):
        if len(self.y) == 1:
            return True
        elif self.impurity is not None and self.impurity < 0.001:
            return True
        else:
            return False


class DT:

    def __init__(self, X, y) -> None:
        self.root = Node(X=X, y=y)

    def fit(self):
        print("Started training...\n")
        self.root.split()
        print("Training succesful!\n")

    def predict(self, X):

        result = []
        for x in X:
            result.append(self.get_prediction(x, self.root))

        return result

    def get_prediction(self, x, node):

        if not node.is_leaf_node():

            if x[node.feature] > node.threshold and node.right:
                return self.get_prediction(x, node.right)
            elif x[node.feature] < node.threshold and node.left:
                return self.get_prediction(x, node.left)

        return sorted([(key, value) for key, value in Counter(node.y).items()], reverse=True, key=lambda x: x[1])[0][0]


if __name__ == "__main__":

    # Test
    X, y = load_iris()
    dt = DT(X, y)
    dt.fit()
    predicted = np.array(dt.predict(X))
    print("Accuracy score")
    print(accuracy_score(y, predicted))
