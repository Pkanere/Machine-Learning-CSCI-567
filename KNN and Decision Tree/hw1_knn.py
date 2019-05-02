from __future__ import division, print_function

from typing import List

import numpy as np
import scipy

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################

class KNN:

    def __init__(self, k: int, distance_function):
        self.k = k
        self.distance_function = distance_function

    #TODO: save features and lable to self
    def train(self, features: List[List[float]], labels: List[int]):
        # features: List[List[float]] a list of points
        # labels: List[int] labels of features
        #raise NotImplementedError
        self.features=features
        self.labels=labels
        
    #TODO: predict labels of a list of points
    def predict(self, features: List[List[float]]) -> List[int]:
        # features: List[List[float]] a list of points
        # return: List[int] a list of predicted labels
        #raise NotImplementedError
        predict_label=[]
        for point in features:
            #print (point)
            neighbours = self.get_k_neighbors(point)
            max_class={}
            for x in neighbours:
                if x in max_class:
                    max_class[x]+=1
                else:
                    max_class[x]=1
            #print(max_class)
            predict_label.append(max(max_class,key = max_class.get))
        return predict_label

    #TODO: find KNN of one point
    def get_k_neighbors(self, point: List[float]) -> List[int]:
        # point: List[float] one example
        # return: List[int] labels of K nearest neighbor
        #raise NotImplementedError
        
        distance = []
        #print(self.features)
        for x in range(len(self.features)):
                #print(self.features[x])
                dst=self.distance_function(self.features[x],point)
                distance.append([self.labels[x],dst])
        #print (distance)
        distance.sort(key=lambda x: x[1])    
        neighbours=[]
        for x in range(self.k):
            neighbours.append(distance[x][0])
        return neighbours


if __name__ == '__main__':
    print(np.__version__)
    print(scipy.__version__)
