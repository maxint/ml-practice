#! /usr/bin/env python
# coding: utf-8

import math


class Tree(object):
    """
    :type parent: Tree
    :type children: list[Tree]
    """
    def __init__(self, parent=None):
        self.parent = parent
        self.children = []
        self.label = None
        self.splitFeature = None
        self.splitFeatureValue = None


def dataToDistribution(data):
    allLabels = [label for (label, _) in data]
    possibleLabels = set(allLabels)
    numEntries = len(allLabels)
    return [float(allLabels.count(label)) / numEntries for label in possibleLabels]


def entropy(dist):
    return -sum([p*math.log(p, 2) for p in dist])


def splitData(data, featureIndex):
    attrValues = set([point[featureIndex] for (_, point) in data])
    for aValue in attrValues:
        yield [(label, point) for (label, point) in data if point[featureIndex] == aValue]


def gain(data, featureIndex):
    entropyGain = entropy(dataToDistribution(data))
    for dataSubset in splitData(data, featureIndex):
        entropyGain -= entropy(dataToDistribution(dataSubset))
    return entropyGain


def homogeneous(data):
    return len(set([label for (label, _) in data])) <= 1


def majorityVote(data):
    labels = [label for (label, point) in data]
    choice = max(set(labels), key=labels.count)
    return choice


def buildDecisionTree(data, root, remainingFeatures):
    """
    :type root: Tree
    :rtype: Tree
    """
    if homogeneous(data):
        root.label = data[0][0]
        return root

    if len(remainingFeatures) == 0:
        root.label = majorityVote(data)
        return root

    remainingGains = [gain(data, aFeature) for aFeature in remainingFeatures]
    bestFeatureIdx, bestFeature = max(enumerate(remainingFeatures), key=lambda d: remainingGains[d[0]])
    if remainingGains[bestFeatureIdx] == 0:
        root.label = majorityVote(data)
        return root

    root.splitFeature = bestFeature

    childRemainingFeatures = remainingFeatures - {bestFeature}
    for dataSubset in splitData(data, bestFeature):
        aChild = Tree(parent=root)
        aChild.splitFeatureValue = dataSubset[0][1][bestFeature]
        root.children.append(aChild)
        buildDecisionTree(dataSubset, aChild, childRemainingFeatures)

    return root


def decisionTree(data):
    return buildDecisionTree(data, Tree(), set(range(len(data[0][1]))))


def classify(tree, point):
    """
    :type tree: Tree
    :type point: list
    """
    if len(tree.children) == 0:
        return tree.label
    else:
        for child in tree.children:
            if child.splitFeatureValue == point[tree.splitFeature]:
                return classify(child, point)
        raise Exception("Classify is not able to handle noisy data. Use classify2 instead.")


def testClassification(data, tree):
    actualLabels = [label for label, _ in data]
    predictedLabels = [classify(tree, point) for _, point in data]
    correctLabels = [a == b for a, b in zip(actualLabels, predictedLabels)]
    return float(sum(correctLabels)) / len(actualLabels)


def printDecisionTree(root, indent=''):
    """
    :type root: Tree
    """
    if len(root.children) == 0:
        print '%s%s, %s' % (indent, root.splitFeatureValue, root.label)
    else:
        if indent == '':
            print '%s%s' % (indent, root.splitFeature)
        else:
            print '%s%s, %s' % (indent, root.splitFeatureValue, root.splitFeature)

        for child in root.children:
            printDecisionTree(child, indent + '|-- ')


if __name__ == '__main__':
    with open('../data/house-votes-1984.txt') as f:
        data = [line.strip().split(',') for line in f.readlines()]
    data = [(x[0], x[1:]) for x in data]

    cleanData = [x for x in data if '?' not in x[1]]
    noisyData = [x for x in data if '?' in x[1]]

    print 'Train data:', len(cleanData)
    print 'Test data:', len(noisyData)
    print 'Feature numbers:', len(cleanData[0][1])

    tree = decisionTree(cleanData)
    printDecisionTree(tree)
    print 'Classification accuracy:', testClassification(noisyData, tree)