from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score
from random import randint
import numpy as np

def runTest(data, labels):
    alpha = 0.0001
    activation = "relu"
    solver = "adam"
    hidden_layer_sizes = []
    for i in range(1, 11):
        sublist = []
        for j in range(1, 11):
            t = [j] * i
            sublist.append(tuple(t))
        hidden_layer_sizes.append(sublist)
    #print(hidden_layer_sizes)
    


def correlation(data, labels):
    """
    Return the correlation between a label assignment and the predictions of
    the classifier
    Args:
      data: A list of datapoints
      labels: The list of labels we correlate against (+1 / -1)
    """

    assert len(data) == len(labels), \
        "Data and labels must be the same size %i vs %i" % \
        (len(data), len(labels))

    assert all(x == 1 or x == -1 for x in labels), "Labels must be binary"

    # DONE

    predicted = data

    return float(np.dot(predicted, labels)) / float(len(data))


def Rademacher_Coeff(number, random_seed=0):
    """
    Generate a desired number of coin tosses with +1/-1 outcomes.
    Args:
      number: The number of coin tosses to perform
      random_seed: The random seed to use
    """
    if random_seed != 0:
        seed(random_seed)

    return [randint(0, 1) * 2 - 1 for x in range(number)]

iris = datasets.load_breast_cancer()
#print(len(iris.data))
#print(len(iris.target))
clf = MLPClassifier(solver='lbfgs', alpha=1e-1, hidden_layer_sizes=(1, 1), random_state=1)
clf.fit(iris.data[:-200], iris.target[:-200])

prediction = clf.predict(iris.data[:-200])
sigmalist = Rademacher_Coeff(369)
print(correlation(prediction, sigmalist))

prediction = clf.predict(iris.data[-200:])
#print(prediction)
print(accuracy_score(iris.target[-200:], prediction))
sigmalist = Rademacher_Coeff(200)
print(correlation(prediction, sigmalist))
runTest(iris.data[:-200], prediction)