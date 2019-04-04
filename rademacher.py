# One extra day used for the extra credit in this assignment. Remaining extra days is 4

from random import randint, seed
from collections import defaultdict
from math import atan, sin, cos, pi, sqrt

import numpy as np
from numpy import array
from numpy.linalg import norm

from itertools import combinations
from sklearn.neighbors import KDTree

from bst import BST

kSIMPLE_DATA = [(1., 1.), (2., 2.), (3., 0.), (4., 2.)]


class Classifier:
    def correlation(self, data, labels):
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

        predicted = [1 if(self.classify(d)) else -1 for d in data]

        return float(np.dot(predicted, labels)) / float(len(data))


class PlaneHypothesis(Classifier):
    """
    A class that represents a decision boundary.
    """

    def __init__(self, x, y, b):
        """
        Provide the definition of the decision boundary's normal vector
        Args:
          x: First dimension
          y: Second dimension
          b: Bias term
        """
        self._vector = array([x, y])
        self._bias = b

    def __call__(self, point):
        return self._vector.dot(point) - self._bias

    def classify(self, point):
        return self(point) >= 0

    def __str__(self):
        return "x: x_0 * %0.2f + x_1 * %0.2f >= %f" % \
            (self._vector[0], self._vector[1], self._bias)


class OriginPlaneHypothesis(PlaneHypothesis):
    """
    A class that represents a decision boundary that must pass through the
    origin.
    """

    def __init__(self, x, y):
        """
        Create a decision boundary by specifying the normal vector to the
        decision plane.
        Args:
          x: First dimension
          y: Second dimension
        """
        PlaneHypothesis.__init__(self, x, y, 0)


class AxisAlignedRectangle(Classifier):
    """
    A class that represents a hypothesis where everything within a rectangle
    (inclusive of the boundary) is positive and everything else is negative.
    """

    def __init__(self, start_x, start_y, end_x, end_y):
        """
        Create an axis-aligned rectangle classifier.  Returns true for any
        points inside the rectangle (including the boundary)
        Args:
          start_x: Left position
          start_y: Bottom position
          end_x: Right position
          end_y: Top position
        """
        assert end_x >= start_x, "Cannot have negative length (%f vs. %f)" % \
            (end_x, start_x)
        assert end_y >= start_y, "Cannot have negative height (%f vs. %f)" % \
            (end_y, start_y)

        self._x1 = start_x
        self._y1 = start_y
        self._x2 = end_x
        self._y2 = end_y

    def classify(self, point):
        """
        Classify a data point
        Args:
          point: The point to classify
        """
        return (point[0] >= self._x1 and point[0] <= self._x2) and \
            (point[1] >= self._y1 and point[1] <= self._y2)

    def __str__(self):
        return "(%0.2f, %0.2f) -> (%0.2f, %0.2f)" % \
            (self._x1, self._y1, self._x2, self._y2)


class ConstantClassifier(Classifier):
    """
    A classifier that always returns true
    """

    def classify(self, point):
        return True


def constant_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over the single constant
    hypothesis possible.
    Args:
      dataset: The dataset to use to generate hypotheses
    """
    yield ConstantClassifier()


def origin_plane_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over hypotheses that result in
    distinct classifications of those points.
    Classifiers are represented as a vector.  The classification decision is
    the sign of the dot product between an input point and the classifier.
    Args:
      dataset: The dataset to use to generate hypotheses
    """

    # DONE

    copyDataset = np.array(dataset)

    theta = np.multiply(np.arctan2(
        copyDataset[:, 1], copyDataset[:, 0]), 180 / np.pi)

    theta.sort()

    hypothesesTheta = list()

    for idx in range(len(theta) - 1):

        if (theta[idx] != theta[idx + 1]):
            meanTheta = (theta[idx] + theta[idx + 1]) / 2

            hypothesesTheta.append(meanTheta)

    hypothesesTheta.append(theta[-1] + np.spacing(np.single(1)))

    hypotheses = np.zeros((len(2 * hypothesesTheta), 2))
    idx1 = 0
    for idx2, theta in enumerate(hypothesesTheta, 0):

        hypotheses[idx1][0] = -1
        hypotheses[idx1][1] = np.tan(hypothesesTheta[idx2])
        hypotheses[idx1 + 1][0] = 1
        hypotheses[idx1 + 1][1] = -np.tan(hypothesesTheta[idx2])

        idx1 += 2

    for h in hypotheses:
        yield OriginPlaneHypothesis(h[0], h[1])


def plane_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over hypotheses that result in
    distinct classifications of those points.
    Classifiers are represented as a vector and a bias.  The classification
    decision is the sign of the dot product between an input point and the
    classifier plus a bias.
    Args:
      dataset: The dataset to use to generate hypotheses
    """

    # Done

    copyDataset = np.array(dataset)

    while len(copyDataset) != 1:

        origin = copyDataset[0, :]

        newData = np.array(dataset)[:len(copyDataset)] - \
            np.array([origin] * len(copyDataset))

        theta = np.multiply(np.arctan2(
            newData[:, 1], newData[:, 0]), 180 / np.pi)

        np.unique(theta)

        meanTheta = [(theta[idx] + theta[idx + 1]) /
                     2 for idx in range(len(theta) - 1)]

        meanTheta.append(theta[-1] + np.spacing(np.single(1)))

        for theta in meanTheta:
            hypothesis = [[-np.tan(theta), 1, origin[0] * np.tan(theta) - origin[1]],
                          [np.tan(theta), -1, origin[1] - origin[0] * np.tan(theta)]]
            for h in hypothesis:
                yield PlaneHypothesis(h[0], h[1], h[2])

        copyDataset = np.delete(copyDataset, 0, 0)

    return


def axis_aligned_hypotheses(dataset):
    """
    Given a dataset in R2, return an iterator over hypotheses that result in
    distinct classifications of those points.
    Classifiers are axis-aligned rectangles
    Args:
      dataset: The dataset to use to generate hypotheses
    """

    # DONE

    xTree = BST()
    yTree = BST()

    for pts in dataset:
        xTree.insert(pts)
        yTree.insert((pts[1], pts[0]))

    yield AxisAlignedRectangle(float('inf'), float('inf'), float('inf'), float('inf'))

    hypotheses = list()

    for numPoints in range(1, len(dataset) + 1):

        hypothesesCombinations = combinations(dataset, numPoints)
        for h in hypothesesCombinations:
            xMin, xMax = min(h, key=lambda x: x[0])[
                0], max(h, key=lambda x: x[0])[0]
            yMin, yMax = min(h, key=lambda x: x[1])[
                1], max(h, key=lambda x: x[1])[1]

            rangeX = [x.key for x in xTree.range((xMin, yMin), (xMax, yMax))]
            rangeY = [(y.key[1], y.key[0])
                      for y in yTree.range((yMin, xMin), (yMax, xMax))]

            if len(set(rangeX) & set(rangeY)) == numPoints:
                if AxisAlignedRectangle(xMin, yMin, xMax, yMax) not in hypotheses:
                    hypotheses.append(
                        AxisAlignedRectangle(xMin, yMin, xMax, yMax))

    for h in hypotheses:
        yield h


def coin_tosses(number, random_seed=0):
    """
    Generate a desired number of coin tosses with +1/-1 outcomes.
    Args:
      number: The number of coin tosses to perform
      random_seed: The random seed to use
    """
    if random_seed != 0:
        seed(random_seed)

    return [randint(0, 1) * 2 - 1 for x in range(number)]


def rademacher_estimate(dataset, hypothesis_generator, num_samples=500,
                        random_seed=0):
    """
    Given a dataset, estimate the rademacher complexity
    Args:
      dataset: a sequence of examples that can be handled by the hypotheses
      generated by the hypothesis_generator
      hypothesis_generator: a function that generates an iterator over
      hypotheses given a dataset
      num_samples: the number of samples to use in estimating the Rademacher
      correlation
    """

    total = list()

    for ii in range(num_samples):
        if random_seed != 0:
            rademacher = coin_tosses(len(dataset), random_seed + ii)
        else:
            rademacher = coin_tosses(len(dataset))

        total.append(max([h.correlation(dataset, rademacher)
                          for h in list(hypothesis_generator(dataset))]))

    return sum(total) / num_samples


if __name__ == "__main__":
    print("Rademacher correlation of constant classifier %f" %
          rademacher_estimate(kSIMPLE_DATA, constant_hypotheses))
    '''print("Rademacher correlation of rectangle classifier %f" %
          rademacher_estimate(kSIMPLE_DATA, axis_aligned_hypotheses))'''
    print("Rademacher correlation of origin centered plane classifier %f" %
          rademacher_estimate(kSIMPLE_DATA, origin_plane_hypotheses))
    print("Rademacher correlation of arbitrary plane classifier %f" %
          rademacher_estimate(kSIMPLE_DATA, plane_hypotheses))