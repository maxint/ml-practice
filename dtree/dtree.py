#! /usr/bin/env python
# coding: utf-8

"""
https://github.com/j2kun/decision-trees/blob/master/decision-tree.py
http://jeremykun.com/2012/10/08/decision-trees-and-political-party-classification/
"""

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
        self.classCounts = None
        self.splitFeature = None
        self.splitFeatureValue = None


def printDecisionTree(root, indent=''):
    """
    :type root: Tree
    """
    if len(root.children) == 0:
        print '%s%s, %s %s' % (indent, root.splitFeatureValue, root.label, root.classCounts)
    else:
        if indent == '':
            print '%s%s' % (indent, root.splitFeature)
        else:
            print '%s%s, %s' % (indent, root.splitFeatureValue, root.splitFeature)

        for child in root.children:
            printDecisionTree(child, indent + '|-- ')


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


def majorityVote(data, node):
    """
    :type node: Tree
    :rtype: Tree
    """
    labels = [label for (label, point) in data]
    node.label = max(set(labels), key=labels.count)
    node.classCounts = dict([(label, labels.count(label)) for label in set(labels)])
    return node


def buildDecisionTree(data, root, remainingFeatures):
    """
    :type root: Tree
    :rtype: Tree
    """
    if homogeneous(data):
        root.label = data[0][0]
        root.classCounts = {root.label: len(data)}
        return root

    if len(remainingFeatures) == 0:
        return majorityVote(data, root)

    remainingGains = [gain(data, aFeature) for aFeature in remainingFeatures]
    bestFeatureIdx, bestFeature = max(enumerate(remainingFeatures), key=lambda d: remainingGains[d[0]])
    if remainingGains[bestFeatureIdx] == 0:
        return majorityVote(data, root)

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


def dictionarySum(*dicts):
    """
    :type dicts: list[dict]
    :return:
    """
    sumDict = {}
    for aDict in dicts:
        for key, val in aDict.iteritems():
            if key in sumDict:
                sumDict[key] += val
            else:
                sumDict[key] = val
    return sumDict


def classifyNoisy(tree, point):
    """
    :type tree: Tree
    :type point: list
    """
    if len(tree.children) == 0:
        return tree.classCounts
    elif point[tree.splitFeature] == '?':
        dicts = [classifyNoisy(child, point) for child in tree.children]
        return dictionarySum(*dicts)
    else:
        for child in tree.children:
            if child.splitFeatureValue == point[tree.splitFeature]:
                return classifyNoisy(child, point)
        raise Exception("Can not been happened")


def classify2(tree, point):
    counts = classifyNoisy(tree, point)
    if len(counts) == 1:
        return counts.keys()[0]
    else:
        return max(counts.keys(), key=lambda k: counts[k])


def testClassification(data, tree, classifier=classify2):
    actualLabels = [label for label, _ in data]
    predictedLabels = [classifier(tree, point) for _, point in data]
    correctLabels = [a == b for a, b in zip(actualLabels, predictedLabels)]
    return float(sum(correctLabels)) / len(actualLabels)


def testTreeSize(noisyData, cleanData):
    import random

    for i in range(1, len(cleanData)):
        tree = decisionTree(random.sample(cleanData, i))
        print i, str(testClassification(noisyData, tree))


if __name__ == '__main__':
    with open('../data/house-votes-1984.txt') as f:
        data = [line.strip().split(',') for line in f.readlines()]
    data = [(x[0], x[1:]) for x in data]

    cleanData = [x for x in data if '?' not in x[1]]
    noisyData = [x for x in data if '?' in x[1]]
    featureNumber = len(data[0][1])

    print 'Train data:', len(cleanData)
    print 'Test data:', len(noisyData)
    print 'Feature numbers:', featureNumber

    testTreeSize(noisyData, cleanData)

    tree = decisionTree(cleanData)
    printDecisionTree(tree)

    # Classification accuracy: 0.935960591133
    print 'Classification accuracy:', testClassification(noisyData, tree)

    print classify(tree, ['y' for _ in range(featureNumber)])  # R
    print classify(tree, ['n' for _ in range(featureNumber)])  # D
