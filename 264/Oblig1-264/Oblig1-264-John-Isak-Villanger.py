import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class Node:
    def __init__(self, dominant_label, feature=None, brink=None, left_child=None, right_child=None):
        self.dominant_label = dominant_label  # label in leaf nodes.
        self.feature = feature
        self.brink = brink
        self.left_child = left_child
        self.right_child = right_child
        self.leaf = left_child is None and right_child is None


def find_dominant_label(y):
    if np.sum(y) > y.size / 2:
        return 1
    return 0


def calculate_entropy_help(y, gini):
    size = y.size
    n_ones = np.sum(y)
    n_zeros = size - n_ones
    if n_ones == y.size or n_ones == 0:
        return 0
    if gini:
        return 1 - ((n_ones / size) ** 2 + (n_zeros / size) ** 2)
    return -(n_ones / size) * np.log2(n_ones / size) - (n_zeros / size) * np.log2(n_zeros / size)


def calculate_entropy(feature, threshold, X, y, gini):
    x_left, x_right, y_left, y_right = marginalize(feature, threshold, X, y)
    e1 = calculate_entropy_help(y_left, gini)
    e2 = calculate_entropy_help(y_right, gini)
    entropy = (y_left.size / y.size) * e1 + (y_right.size / y.size) * e2
    return x_left, x_right, y_left, y_right, entropy


def marginalize(feature, threshold, X, y):
    arr_filter = X[:, feature] < threshold
    arr_filter_negated = [not val for val in arr_filter]
    x_left = X[arr_filter]
    x_right = X[arr_filter_negated]
    y_left = y[arr_filter]
    y_right = y[arr_filter_negated]
    return x_left, x_right, y_left, y_right


def check_identical_feature_values(x):
    for _x in x:
        if not np.all(_x == _x[0]):
            return False
    return True


def learn(X, y, gini):
    if np.sum(y) == 0 or np.sum(y) == y.size:
        if np.sum(y) == 0:
            return Node(0)
        return Node(1)
    elif check_identical_feature_values(X):
        return Node(find_dominant_label(y))

    smallest = 1
    for i in range(X[0].size):
        sorted_x = np.sort(X[:, i])
        for j in range(sorted_x.size - 1):
            threshold_ = (sorted_x[j] + sorted_x[j + 1]) / 2
            x_left_, x_right_, y_left_, y_right_, entropy = calculate_entropy(i, threshold_, X, y, gini)
            if entropy < smallest:
                smallest = entropy
                x_left = x_left_
                x_right = x_right_
                y_left = y_left_
                y_right = y_right_
                feature = i
                threshold = threshold_

    left_child = learn(x_left, y_left, gini)
    right_child = learn(x_right, y_right, gini)
    return Node(find_dominant_label(y), feature, threshold, left_child, right_child)


def predict(node: Node, x):
    if node.leaf:
        return node.dominant_label
    if x[node.feature] < node.brink:
        return predict(node.left_child, x)
    return predict(node.right_child, x)


def predict_all(node, X):
    size = np.shape(X)[0]
    predictions = np.empty(size)
    for i in range(size):
        predictions[i] = predict(node, X[i])
    return predictions


def prune(node: Node, X, y):
    if not (node.left_child.leaf and node.right_child.leaf):
        x_left, x_right, y_left, y_right = marginalize(node.feature, node.brink, X, y)
        if not node.left_child.leaf:
            prune(node.left_child, x_left, y_left)
        if not node.right_child.leaf:
            prune(node.right_child, x_right, y_right)

    predictions = predict_all(node, X)
    n_right_predictions = np.count_nonzero(y == predictions)
    n_right_majority_label = np.count_nonzero(y == node.dominant_label)
    if n_right_majority_label >= n_right_predictions:
        node.leaf = True
        node.left_child = None
        node.right_child = None


def find_tree_size(node: Node):
    return 1 if node.leaf else 1 + find_tree_size(node.left_child) + find_tree_size(node.right_child)


def find_accuracy(root: Node, X, y, datatype, pruning):
    predictions = predict_all(root, X)
    n_right_predictions = np.count_nonzero(y == predictions)
    fake_misclassified = np.count_nonzero((y == 0) & (predictions == 1))
    fake_notes = y.size - np.sum(y)
    print("Accuracy on", datatype, "data", pruning, "pruning:", n_right_predictions / y.size)
    print("Misclassified fake notes as positive", fake_misclassified, "of", fake_notes, "fake notes \n")


def main():
    data = np.genfromtxt('data_banknote_authentication.txt', delimiter=',')
    X = data[:, 0:4]
    y = data[:, 4:]
    y = y.flatten().astype(int)

    seed = 365  # fixed seed for reproducibility
    x_train_val, x_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=seed)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25, shuffle=True,
                                                      random_state=seed)
    gini = False

    root = learn(x_train, y_train, gini)
    # SanityCheck
    find_accuracy(root, x_train, y_train, "training", "before")
    find_accuracy(root, x_val, y_val, "validation", "before")

    print("Size before pruning ", find_tree_size(root))
    debug_point = 1
    prune(root, x_val, y_val)
    print("Size after pruning ", find_tree_size(root),"\n")

    find_accuracy(root, x_train, y_train, "training", "after")
    find_accuracy(root, x_val, y_val, "validation", "after")
    find_accuracy(root, x_test, y_test, "test", "after")

    debug_point = 2

    skTree = DecisionTreeClassifier()
    skTree.fit(x_train, y_train)
    print("skTree accuracy on validation data", skTree.score(x_val, y_val))
    print("skTree accuracy on test data", skTree.score(x_test, y_test))


main()
