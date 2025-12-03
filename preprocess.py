import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import mutual_info_classif
import time
from collections import Counter
from RandomForest import RandomForest

    #resample the minority class so that the class distribution is even
def smote_resample(x,y,state=42):
    sm = SMOTE(random_state=state)
    print('Original dataset shape %s' % Counter(y))
    x_res, y_res = sm.fit_resample(x, y)
    print('Resampled dataset shape %s' % Counter(y_res))
    return x_res, y_res
    
    #select top n features
def feature_selection(x,y,n,state=42):
    info = mutual_info_classif(x,y, random_state=state)
    series = pd.Series(info,x.columns).sort_values(ascending=False)
    return series.index.to_numpy()[:n]

def preprocess(data,training_size=0.7,test_size=0.3, n_features=20,resample=True,state=42):
    X = data.drop(['ID_code'], axis=1)
    y = X.pop('target')
    # print(X.columns)
    # Sample subset_data of the original data  
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=training_size, random_state=state,stratify=y)

    #Select top n features using feature selection from the training data
    top_features = feature_selection(X_train,y_train,n_features,state)
    
    X_train = X_train[top_features]
    X_test = X_test[top_features]
    
    
    #split test size if only using a small portion, else use the whole x_test
    if training_size + test_size < 1:
        X_test_full=X_test.copy()
        X_test_full['target']=y_test
        X_test=X_test_full.sample(n=int(len(data)*test_size),replace=False,random_state=state)
        y_test = X_test.pop('target')


    #Scale data
    scaler = StandardScaler()
    X_train= pd.DataFrame(scaler.fit_transform(X_train))
    X_test = pd.DataFrame(scaler.transform(X_test))
    
    #resample
    if resample:
        X_train,y_train = smote_resample(X_train,y_train,state)

    return X_train, X_test, y_train, y_test

