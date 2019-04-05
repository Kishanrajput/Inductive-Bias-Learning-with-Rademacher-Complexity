from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score
from random import randint
import numpy as np
import math

def ErrorofH(predictions, labels, D, E, n, m):
    listoferrors = []
    for p in range(D):
        outererrsum = 0
        for i in range(n):
            inf_er_h = math.inf
            for j in range(E):
                sumerror = 0
                for k in range(m):
                    #print(p, j, i, k)
                    l = lossfunction(predictions[p][j][i][k], labels[i][k])
                    sumerror += l
                if inf_er_h > float(sumerror)/m:
                    inf_er_h = float(sumerror)/m

            outererrsum += inf_er_h
        listoferrors.append(float(outererrsum)/n)
    return listoferrors

def ERCofErrorFunction(labels, predictions, sigma, n, m, D, E):
    # n is number of tasks
    # m is number of examples of each task
    # D is number of hypothesis spaces in hypothesis space family
    # E is number of hypothesises in each hypothesis space
    Herrorlist = []
    sup_H = -math.inf
    for p in range(D):
        outersum = 0
        outererrsum = 0
        for i in range(n):
            inf_h = math.inf
            inf_er_h = math.inf
            for j in range(E):
                sum = 0
                sumerror = 0
                for k in range(m):
                    #print(p, j, i, k)
                    l = lossfunction(predictions[p][j][i][k], labels[i][k])
                    sumerror += l
                    sum += sigma[i][k] * l
                if inf_er_h > float(sumerror)/m:
                    inf_er_h = float(sumerror)/m

                if inf_h > float(sum)/m :
                    inf_h = float(sum)/m
                    #print(sum)
            outersum += inf_h
            outererrsum += inf_er_h
        Herrorlist.append(float(outererrsum)/n)
        #print("infimum---", float(outersum)/n)
        #print("Supremum--", sup_H)
        if sup_H < float(outersum)/n:
            sup_H = float(outersum)/n
            #print("assigned")

    return [sup_H, Herrorlist]

def lossfunction(pval, realval):
    if pval == realval:
        return 0
    else:
        return 1



def runTest(data, labels, testdata, testlabels):

    factor = 3 * math.sqrt(math.log(2/0.4)/(2000))

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
    testdata = [testdata] * 10
    testlabels = [testlabels] * 10
    Hhnpredictions = []
    HhnErrPredictions = []
    for i in range(len(hidden_layer_sizes)):
        hnpredictions = []
        hnErrpredictions = []
        for hsize in hidden_layer_sizes[i]:
            clf = MLPClassifier(solver='adam', alpha=alpha, hidden_layer_sizes=hsize, max_iter=5000, random_state=1)
            npredictions = []
            nErrPredictions = []
            for row in range(len(data)):
                clf.fit(data[row], labels[row])
                prediction = clf.predict(data[row])
                npredictions.append(prediction)
                prediction = clf.predict(testdata[row])
                nErrPredictions.append(prediction)
            hnpredictions.append(npredictions)
            hnErrpredictions.append(nErrPredictions)
        Hhnpredictions.append(hnpredictions)
        HhnErrPredictions.append(hnErrpredictions)
    #print(Hhnpredictions)
    sigma = []
    for row in range(len(data)):
        sigmalist = Rademacher_Coeff(len(data[row]))
        #print("sigmalist---", sigmalist)
        sigma.append(sigmalist)

    RofH, Herrorlist = ERCofErrorFunction(labels, Hhnpredictions, sigma, 10, 100, 10, 10)
    print("RofH --- ", round(RofH, 2))
    print("Empirical Error list--- ", Herrorlist)
    print("factor --- ", round(factor,2))
    ActualError = ErrorofH(HhnErrPredictions, testlabels, 10, 10, 10, 50)
    print("Error on Test data --- ", ActualError)
    print("Epsilon -- ", round(RofH+factor, 2))
    for i in range(len(ActualError)):
        if ActualError[i] < (Herrorlist[i] + RofH + factor):
            print(True , ActualError[i], (Herrorlist[i]+RofH+factor), Herrorlist[i])
        else:
            print(False, ActualError[i], (Herrorlist[i]+RofH+factor), Herrorlist[i])



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

print(math.inf)
iris = datasets.load_breast_cancer()
data = iris.data[:100]
#print(len(data))
labels = iris.target[:100]
Testdata = iris.data[100:150]
Testlabels = iris.target[100:150]
runTest(data, labels, Testdata, Testlabels)



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