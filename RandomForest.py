# from RandomForestTreeLinear import RandomForestTree as DecisionTree_linear
from RandomForestTree import RandomForestTree as DecisionTree

import pandas as pd
import numpy as np
from collections import Counter
import multiprocessing
from multiprocessing import Pool,cpu_count,shared_memory
import random
import time

def build_tree(random_state,min_split,n):

        tree = DecisionTree(random_state,min_split)

        #sample indicies
        samped_indicies = bootstrap(random_state)

        #access the memory block
        x_shared = shared_memory.SharedMemory(name=x_meta['name'])
        y_shared = shared_memory.SharedMemory(name=y_meta['name'])
        # print("Size (MB):", round(x_shared.size / (1024 * 1024) + (y_shared.size/(1024*1024)),2))
        #  convert data into nparray using the shpae and dtypes
        x_data = np.ndarray(x_meta['shape'], dtype=np.dtype(x_meta['dtype']), buffer=x_shared.buf)
        y_data = np.ndarray(y_meta['shape'], dtype=np.dtype(y_meta['dtype']), buffer=y_shared.buf)

        start_time = time.perf_counter()    
        tree.fit(x_data,y_data,samped_indicies)        
        end_time = time.perf_counter()

        if verbose: print('seed:',random_state,' finished building tree ',n+1,' it took ', round(end_time-start_time,2),'s.',flush=True)

        #close access after computation is done
        x_shared.close()
        y_shared.close()
        
        return tree
    #return rows/pointers to row
def bootstrap(state):
        index_gen = np.random.default_rng(seed=state)
        return index_gen.integers(0, length, size=length,dtype=np.int32)
        
    #global variables used during tree_build process
def init_global_var(verbose_condition,size,x_dict,y_dict):
    global verbose,length,x_meta, y_meta
    length = size
    verbose=verbose_condition
    x_meta = x_dict
    y_meta = y_dict

class RandomForest:
    def __init__(self,random_state=42,number_of_trees=100,processes=1,min_split=2,verbose=False):
        self.number_of_trees = number_of_trees
        self.random_state=random_state
        self.min_split=min_split
        self.tree_list=list()
        self.verbose=verbose
        # self.is_linear = is_linear

        if processes == -1 or processes >= cpu_count():
            self.processes = cpu_count()
        else:
            self.processes=processes

    def fit(self,x,y):
        # convert x and y to numpy and store them in shared memory, so that each child process doesnt make a copy
        x_shared,x_dict = self.to_shared_memory(x.to_numpy(),'rf_x_data')
        y_shared,y_dict = self.to_shared_memory(y.to_numpy(),'rf_y_data')
        #arguments for build_tree function
        params = self.get_parameters()
        with Pool(processes=self.processes, initializer=init_global_var, initargs=(self.verbose,len(x),x_dict,y_dict)) as p:            
            self.tree_list = p.starmap(build_tree,self.get_parameters())
        #free the allocated memory
        self.close_and_unlink(x_shared)
        self.close_and_unlink(y_shared)
   

    #returns a 3 tuple list of size number_of_trees, 1st tuple containing a random seed for bootstrapping function, 2nd contains min_split, 3rd contains tree number
    #arguments for multiprocess worker(build_tree) function 
    def get_parameters(self):
        params = list()
        random_state = random.Random(self.random_state)        
        for i in range(self.number_of_trees):
            rn = random_state.randint(0,100000)
            params.append((rn,self.min_split,i))
        return params 

    def to_shared_memory(self,x,name):
        #create a memory block of size x
        shm = shared_memory.SharedMemory(name=name,create=True, size=x.nbytes) 
        #format the memory block
        shared_data = np.ndarray(x.shape, dtype=x.dtype, buffer=shm.buf)
        #copy x to memory
        shared_data[:] = x[:]
        return shm ,{'name': name, 'shape': x.shape, 'dtype': x.dtype.str}

    def close_and_unlink(self,x):
        x.close()
        x.unlink()    

    def predict(self,x_test):
        x_test_copy = x_test.copy()
        x_test_copy['__row'] = x_test_copy.index 
        all_predictions = []
        for tree in self.tree_list:
            all_predictions.append(tree.predict(x_test_copy))     
        
        # aggregate predictions by majority voting  
        final_predictions = []
        for i in range(len(x_test)):
            tree_predictions_for_sample = [pred[i] for pred in all_predictions] # gather predictions for the i-th sample from all trees
            final_predictions.append(self._most_common_class(tree_predictions_for_sample))        
        return final_predictions

        # helper function to find the most common class in a list, used in predict()
    def _most_common_class(self,y_predictions):
        return Counter(y_predictions).most_common(1)[0][0]  # returns the most common class label, assuming classification task

