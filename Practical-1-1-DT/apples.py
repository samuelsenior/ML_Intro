#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Greg, Sam, Toshan
"""
import numpy as np
from sklearn import tree


def apples_oranges():
    
    """
    In this example, lets assume we can't see the piece of fruit, but
    we only have access to the weights of the fruit and their texture,
    described as 'bumpy' or 'smooth'. Given some data, can we 'learn'
    which pieces of fruit are apples, and which ones are oranges with any
    degree of confidence? Yes we can!
    """
    
    """
    in features:
        - index 0 : weight of the fruit (g)
        - index 1 : (texture of fruit) binary, 0 = bumpy fruit, 1 = smooth fruit
    in labels :
        - 0 = apple
        - 1 = orange
    """
    
    # here we create our features as a matrix
    X = features = [[140, 1], [130, 1], [150, 0], [170, 0], [145, 1],
                    [155, 0], [152, 1], [175, 0]]
    # here we create our labels as a vector
    y = labels = [0, 0, 1, 1, 0, 1, 0, 1]
    # each index corresponds to the 2 features
    
    # now create our decision tree
    clf = tree.DecisionTreeClassifier()
    # now fit the tree, using our features and labels (supervised learning)
    clf.fit(features, labels)
    
    # now we introduce a new fruit, we only know it's weight and
    # whether it's smooth or bumpy, can we predict what it will be?
    
    # here we have a piece of fruit weighing 125g and 'smooth' texture
    fruit = [125, 1]
    guess = clf.predict([fruit])
    if guess[0] == 0:
        print("We predict an apple!")
    else:
        print("We predict an orange!")
    
    """
    Insert your code here to try examples
    """
    # -------------------------------------
    
    
    # -------------------------------------
    
    return clf


def print_out_tree(clf):
    tree.export_graphviz(clf, out_file='tree.dot') 
    


if __name__ == '__main__':
    atree = apples_oranges()
    print_out_tree(atree)
