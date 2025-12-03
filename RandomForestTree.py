import random
import math
import pandas as pd
from TreeNode import TreeNode as Node 
from collections import Counter
import numpy as np
import time

class RandomForestTree:
    def __init__(self,random_state=42,min_split=2,max_depth=None):
        self.rng =random.Random(random_state)
        self.tree_root=None
        self.min_split=min_split
        self.max_depth = max_depth
        
    def fit(self,x,y,indices):
        self.cols = x.shape[1]
        self.features_per_split = round(math.sqrt((self.cols)))
        self.tree_root = self.build_tree(x,y,indices.astype(np.int32),0)

    def build_tree(self,x,y,indices,depth):
        node,gain = self.best_split(indices,x,y,self.sample_features())     
        # base case
        leaf = self.is_terminal(y,indices,gain,depth)       
        if leaf is not None:
            node.set_leaf(leaf)
            return node
        # masking condition, used to split indices to left and right based on feature and threshold 
        split_condition = x[indices,node.get_feature()] > node.get_threshold() 
        
        # recursion
        node.set_children(
            left=self.build_tree(x,y,indices[~split_condition],depth+1),
            right=self.build_tree(x,y,indices[split_condition],depth+1)
            )
        return node

    #deciding when node is terminal
    def is_terminal(self,y,indices,gain,depth):
        #min_split reached
        if len(indices) <= self.min_split:
            return Counter(y[indices]).most_common(1)[0][0]
        #pure node
        elif gain == 0:
            return y[indices][0]
        #max depth reached
        elif self.max_depth is not None and self.max_depth == depth:
            return Counter(y[indices]).most_common(1)[0][0]
        return None

    # returns the best feature and the best randomly sampled threshold from a random list of features
    def best_split(self,indexes,x,y,random_features):
        #tuple 1 refer to the best feature, 2 to best threshold for that feature, 3 refers to the information gain 
        best_feature = (None,0)
        max_gain = -1
        data_length = len(indexes)

        # transform into bootstrapped indexes
        y_cols = y[indexes]
        feature_entropy = self.calculate_entropy(y_cols)

        # iterate over the list of random features
        for feature in random_features: 
            #get feature on rows x
            x_col = x[indexes,feature]
            unique_vals = np.unique(x_col)

        # sample sqrt(n) unique values to find best threshold
            subset = self.random_unique_values(int(math.sqrt(len(unique_vals))),unique_vals)
        
        # iterate over every unique value of feature x, to find the best threshold to maximize information gain 
            for value in subset:              
                split_condition = x_col > value 
                gain = self.calculate_gain(left=y_cols[~split_condition],right=y_cols[split_condition],feature_entropy=feature_entropy,weight=data_length)
                
                # replace if gain from current threshold is better than previous best
                if gain > max_gain:
                    best_feature = (feature,value)
                    max_gain=gain
            # if best_feature[0] is None:
            #     print("ALERT",len(y_cols), 'has gain', best_feature[1])
        return Node(feature=best_feature[0],threshold=best_feature[1]),max_gain


    # calculate gain, on a split by calculating weighted left and weighted right entropy, then subtracting from feature entropy
    def calculate_gain(self,left,right,feature_entropy,weight):
        weighted_right_entropy = self.calculate_entropy(right) * (len(right)/weight)
        weighted_left_entropy = self.calculate_entropy(left) * (len(left)/weight)
        return feature_entropy - (weighted_left_entropy+weighted_right_entropy)

    #calculates entropy    
    def calculate_entropy(self,data):
        entropy = 0 
        counts = np.unique(data, return_counts=True)[1]
        probability = counts / counts.sum()
        
        for p in probability:
            entropy = entropy - p*math.log2(p)
        return entropy
        # return -(probability * np.log2(probability)).sum()
   
    #randomly sample n unique values, the caller passes n as sqrt n 
    def random_unique_values(self,n,unique_vals):
        rand_gen = random.Random(self.rng.randint(0,100000))
        return rand_gen.sample(list(unique_vals),n)

    # returns features_per_split random features
    def sample_features(self):
        random_feature_gen = random.Random(self.rng.randint(0, 100000)) 
        return random_feature_gen.sample(range(self.cols), k= self.features_per_split)

    def _traverse_tree(self,data,node):
        #base case, append leaf result and row number
        if node.is_leaf(): 
            for i,row in data.iterrows(): 
                self.prediction_results.append((node.get_leaf(),int(row['__row'])))
            return

        threshold = node.get_threshold()
        feature = node.get_feature()
        #recurse right and left if data is not empty
        if len(data) > 0:
            mask = data.iloc[:,feature] > threshold
            self._traverse_tree(data[mask],node.right)
            self._traverse_tree(data[~mask],node.left)
        return 

    def predict(self,x_test):
        self.prediction_results = []
        self._traverse_tree(x_test,self.tree_root)
        result = self.rearrange_results(self.prediction_results)
        self.prediction_results=[]
        # print(result)
        return result

    def rearrange_results(self,results):
        ordered_results = [None]*len(results)
        for result,row in results:
            ordered_results[row]=result
        return ordered_results