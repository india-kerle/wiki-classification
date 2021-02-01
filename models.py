import pandas as pd
from sklearn import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
import numpy as np
import pandas as pd


def run_models(X_train , X_test, y_train):
    '''takes as input vectorised X_train, X_test and y_train and outputs table of evaluation scores, mean f1 score, name of best model based on mean f1 score and list of models trained.'''
    dfs = []
    models = [
              ('LogReg', LogisticRegression()), 
              ('RF', RandomForestClassifier()),
              ('decision_tree', tree.DecisionTreeClassifier()),
              ('GNB', GaussianNB()),
              ('SVM', SGDClassifier())
               ]
    
    results = []
    names = []
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
    target_names = ['1', '0']
    
    for name, model in models:
        kfold = model_selection.KFold(n_splits = 5, shuffle = True, random_state = 2345)
        cv_results = model_selection.cross_validate(model, X_train, y_train, cv = kfold, scoring = scoring)
        print(f'running {name} ...!')
        clf = model.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        
        results.append(cv_results)
        names.append(name)
        this_df = pd.DataFrame(cv_results)
        this_df['model'] = name
        dfs.append(this_df)
        
    final = pd.concat(dfs, ignore_index = True)
    final_mean = final.groupby(['model']).mean()
    best_model = final_mean.index[final_mean["test_f1_weighted"] == final_mean.test_f1_weighted.max()].to_list()
    best_model = ''.join(best_model)

    
    return final, final_mean, best_model, models

def pick_hyperparameters():
    '''outputs dictionaries of different hyperparameters to tune based on best model.'''
 
    LogReg_grid_param = {
    'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
    'C': [0.01, 0.1, 1, 10, 100]
    }

    RF_grid_param = {
    'n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]

    }

    decision_tree_grid_param = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4] 
    }

    GNB_grid_param = {
    'estimator__var_smoothing': [1e-2, 1e-5, 1e-10, 1e-15]
    }

    SVM_grid_param = {
    'alpha': [1e-4, 1e-3, 1e-2], # learning rate
    'max_iter':  [500, 1000],
    'loss': ['hinge', 'log'], 
    'penalty': ['l2', 'l1']
    }    

    return LogReg_grid_param, RF_grid_param, decision_tree_grid_param, GNB_grid_param, SVM_grid_param

def best_tuned_model(best_model, models, X_train, y_train):
    '''takes as input name of best_model based on f2 score, list of models, vectorised X_train and y_train and outputs dataframe of optimal hyperparameters for best model.'''
   
    tuning = ['LogReg_grid_param', 'RF_grid_param', 'decision_tree_grid_param',
             'GNB_grid_param', 'SVM_grid_param']

    param_grid = pick_hyperparameters()[tuning.index(best_model + '_grid_param')]
    for name, model in models:
        if name == best_model:
            print(f'busy tuning {best_model}...')
            clf = GridSearchCV(estimator = model, 
                               param_grid = param_grid,
                               scoring = 'f1_weighted',
                               cv = 5, 
                               verbose = 0)
    
    grid_result = clf.fit(X_train, y_train)   
    print('Best weighted f1 score: ', clf.best_score_)
    best = clf.best_estimator_

    return best  

def predict_test_labels(test, best_toxic_model, best_obscene_model, 
                        X_train, X_test, y_toxic_train, y_obscene_train):
    '''takes as input best model, vectorised X_train and X_test and outputs submission dataframe with predicted labels per text.'''

    class_names = ['toxic', 'obscene']
    submission = pd.DataFrame.from_dict({'id': test['id'],
                                        'comment': test['comment_text']})

    for class_name in class_names:
        if class_name == 'toxic':
            toxic_classifier = best_toxic_model
            toxic_classifier.fit(X_train, y_toxic_train)
            submission['toxic'] = toxic_classifier.predict(X_test)
        if class_name == 'obscene':
            obscene_classifier = best_obscene_model
            obscene_classifier.fit(X_train, y_obscene_train)
            submission['obscene'] = obscene_classifier.predict(X_test)

    return submission