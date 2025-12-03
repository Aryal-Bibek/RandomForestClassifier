Update File path in main.py to the path of the data file

Data can be found on Kaggle: https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data

DATASET IS TOO BIG TO BE INCLUDED HERE!

To use our RandomForestClassifier model, in main.py declare declare a RandomForest object with the following hyper parameter: 

    processses = # number of processes, default is 1
    min_split = #minimum number of examples for the node to be declared terminal, default 2
    random_state = #default is 42, ensures reproducibility
    verbose = # True or False, prints the trees that have finished building, and their time in    console
    number_of_trees = # number of trees, default is 100
    
    
RandomForest functions: 

    fit(x,y): x and y must be dataframes, or the program might not work as intended          
    predict(x_test) #returns a list of class labels corresponding to each row in x_test

Preprocess function has the follow parameters, please use it to clean up your data
    parameters:
	
        training_size = #defaults to 0.7
        test_size = #defaults to 0.3
        n_features = #used for feature selection, selects top_n features using mutual information, default is 20
        resample = #boolean value, will upsample minority class to  1:1
        state = #random state used for all the preprocessing tasks
        
score function in main.py takes in the following params:

    model = String name of the model
    predictions = output of some prediction model
    actual_result = y_test
	score(model,predictions,actual_result): # prints accuracy,precision,recall,f1score and a confusion matrix
