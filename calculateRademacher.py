from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score
from random import randint
import numpy as np

def ERCofErrorFunction(labels, predictions, sigma, n, m, D, E):
    Sup_H = 0
    # n is number of tasks
    # m is number of examples of each task
    # D is number of hypothesis spaces in hypothesis space family
    # E is number of hypothesises in each hypothesis space
    for p in range(D):
        outersum = 0
        for i in range(n):
            inf_h = math.inf
            for j in range(E):
                sum = 0
                for k in range(m):
                    sum += sigma[i][k] * lossfunction(predictions[p][j][i][k], labels[i][k])
                if inf_h > sum/m :
                    inf_h = sum/m
            outsum += inf_h
        if sup_H < outsum/n:
            sup_H = outsum/n
    return sup_H

def lossfunction(pval, realval):
    if pval == realval:
        return 0
    else:
        return 1



def runTest(data, labels):
    alpha = 0.0001

    hidden_layer_sizes = []
    for i in range(1, 11):
        sublist = []
        for j in range(1, 11):
            t = [j] * i
            sublist.append(tuple(t))
        hidden_layer_sizes.append(sublist)
    data = [data] * 10
    labels = [labels] * 10
    Hhnpredictions = []
    for i in range(len(hidden_layer_sizes)):
        hnpredictions = []
        for hsize in i:
            clf = MLPClassifier(solver='adam', alpha=alpha, hidden_layer_sizes=hsize, random_state=1)
            npredictions = []
            for row in range(len(data)):
                clf.fit(data[row], labels[row])
                prediction = clf.predict(data[row])
            npredictions.append(prediction)
        hnpredictions.append(npredictions)
    Hhnpredictions.append(hnpredictions)

    sigma = []
    for row in range(len(data)):
        sigmalist = Rademacher_Coeff(len(data[row]))
        sigma.append(sigmalist)

    print(ERCofErrorFunction(labels, Hhnpredictions, sigma, 10, 100, 10, 10))
    


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
data = iris.data[:100]
#print(len(data))
labels = iris.target[:100]
runTest(data, labels)



'''
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
runTest(iris.data[:-200], prediction)'''