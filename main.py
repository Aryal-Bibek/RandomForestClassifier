from RandomForest import RandomForest
import time, platform
from preprocess import preprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import pandas as pd
import multiprocessing


def fitting_time(classifier,x,y):
    start = time.perf_counter()
    classifier.fit(x,y)
    end = time.perf_counter()
    return round(end - start,2)

def score(model,predictions,actual_result):
    accuracy = round(accuracy_score(actual_result, predictions)*100,2)
    precision =round(precision_score(actual_result, predictions)*100,2)
    recall = round(recall_score(actual_result, predictions)*100,2)
    f1 = round(f1_score(actual_result, predictions)*100,2)
    print(key,'has accuracy of ',accuracy, '%, precision of',precision,'%, recall of',recall,'% and f1 score of',f1,'%\n')
    print((confusion_matrix(actual_result, predictions)),'\n')



if __name__ == "__main__":
    
    #set start should not be fork, it will exceed memory at initiailization
    if platform.system() == 'Windows':
        multiprocessing.set_start_method('spawn')
    else:
        multiprocessing.set_start_method('forkserver')
    
    file_path = 'data/train.csv'
    data = pd.read_csv(file_path)

    # number of trees
    trees =  100
    #random_state
    r_state=500
    #number of process, if (p>cpu_cores || p == -1) then max cores, else p
    p= 10
    
    x,x_test,y,y_test = preprocess(data,training_size=0.7,test_size=0.3, n_features=25,resample=True,state=r_state)
    
    #our implementation, parameter verbose prints the fitting time, and the random state of each tree if true 
    random_forest = RandomForest(processes=p,number_of_trees=trees,random_state=r_state,verbose=True)
    
    #library forest
    library_forest = RandomForestClassifier(criterion='entropy',random_state=r_state,n_jobs=p,n_estimators=trees)
    
    #mlp neural network 
    neural_network= MLPClassifier(hidden_layer_sizes=(100, 50),random_state=r_state, max_iter=1000)
        
    #xgboost classifier
    xgboost = XGBClassifier(objective='binary:logistic', n_estimators=trees, learning_rate=0.1, max_depth=5, random_state=r_state, n_jobs=p, eval_metric='logloss')

    
    models ={
        'Random Forest (our) with '+str(trees)+' trees':random_forest,
        'Random Forest (library) with '+str(trees)+' trees:':library_forest,
        'MLP Neural Network Classifier':neural_network,
         'XGBoost Classifier':xgboost
    }
    for key, model in models.items():
            fit_time = fitting_time(model,x,y)
            minutes = int(fit_time//60)
            seconds = round(fit_time%60,2)
            print("Total fitting time for", key, minutes,'minutes,',seconds, "seconds \n")
            score(key,model.predict(x_test),y_test)

    print('Training Data Size:',len(x),', Training Data Columns:',len(x.columns),', with size',round(
        (x.memory_usage(index=False, deep=True).sum() / (1024 ** 2)) + (y.memory_usage(index=False, deep=True) / (1024 ** 2))
        ,2),'MB')
