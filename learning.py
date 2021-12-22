import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import pickle
import itertools
import os

class Learning:
    
    X = []
    y = []
    flair = []
    param = {}
    
    def __init__(self, X = [], y = [], flair = []):
        self.X = X
        self.y = y
        self.flair = flair
        
        
    # CONFUSION MATRIX, USED TO PLOT CLASSIFIER PREDICTION
    def __plotConfusionMatrix(self,
                            cm, 
                            classes,
                            normalize = True,
                            title = 'Confusion matrix',
                            cmap = plt.cm.Blues):

        if normalize:
            cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
            print("Confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        fig, ax = plt.subplots()
        plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation = 45)
        plt.yticks(tick_marks, classes)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        st.pyplot(fig)
        # plt.show()
        

    # CLASSIFIER: XGBOOST
    # ALGORITHM: SOFTMAX MULTI-CLASS
    # RESULT: PREDICTED PROBABILITY OF EACH DATA POINT BELONGING TO EACH CLASS
    # SOFTMAX: COMPUTE EXP OF INPUT VECTOR TO NORMALIZE DATASET INTO A PROBABILISTIC DISTRIBUTION WITH VALUES SUMMING TO ONE.
    # SOFTMAX: GOOD FOR MULTI-DIMENSIONAL CLASSIFICATION, INSTEAD OF BINARY CLASSIFICATION.
        
    def setXGBLambda(self, lambda_xgb):
        self.param['objective'] = 'multi:softprob'
        self.param['max_depth'] = 3 # depth of tree, default: 6. As value increases: overfit.
        self.param['num_class'] = len(np.unique(self.y))
        self.param['eta'] = lambda_xgb # learning rate, default: 0.3
        

    # TRAIN WITH K-FOLD STRATIFIED VALIDATION
    def trainStratified(self,
                        models = xgb.XGBClassifier(**param), 
                        nsplits = 5, 
                        confusion = True):
        
        k_fold = StratifiedShuffleSplit(n_splits = nsplits)
        
        labels = np.unique(self.flair)
        labels_arr = np.asarray(labels)
        np.savetxt("./data/model/labels.csv", labels, fmt = '%s', delimiter = ',')
                
        count_vectorizer = CountVectorizer(analyzer = "word")
        fig_i = 0
        i = 0
        
        # STRATIFIED SPLIT
        for train, test in k_fold.split(self.X, self.y):
            
            i += 1
            
            # VECTORIZATION WITH COUNT AND TF-IDF
            
            X_train, X_test, y_train, y_test = self.X[train], self.X[test], self.y[train], self.y[test]
                            
            # BAG OF WORDS METHOD: LEARN VOCABULARY
            # RETURN DOCUMENT-TERM MATRIX (scipy.sparse MATRIX) IN FORM: (#BODY, #WORD -> #OCCURRENCES)
            X_train = count_vectorizer.fit_transform(X_train.ravel())
            X_test = count_vectorizer.transform(X_test.ravel())
            
            filename_vect = './data/model/count_vectorizer_pickle_' + str(i) + '.dat'
            os.makedirs(os.path.dirname(filename_vect), exist_ok = True)
            with open('./data/model/count_vectorizer_pickle_' + str(i) + '.dat', 'wb') as f: 
                pickle.dump(count_vectorizer, f)
                            
            probs = np.ones((len(y_test), len(np.unique(self.y))))
                                
            # XGB ACCEPTS ONLY DMatrix DATA TYPE
            xgb_train = xgb.DMatrix(X_train, label = y_train)
            xgb_test = xgb.DMatrix(X_test, label = y_test)
                
            # WHAT TO DISPLAY DURING EXECUTION
            watchlist = [(xgb_train, 'train'), (xgb_test, 'test')]
                
            # NUMBER OF ITERATIONS FOR EACH EXECUTION
            num_round = 45 # found by "trial and error"
                
            # TRAINING
            bst = xgb.train(self.param, xgb_train, num_round, watchlist)
                
            # PREDICTION
            preds = bst.predict(xgb_test)
            probs = np.multiply(probs, preds)
            preds = np.array([np.argmax(prob) for prob in preds])
            
            filename_model = './data/model/mbti_pickle_' + str(i) + '.dat'
            os.makedirs(os.path.dirname(filename_model), exist_ok = True)
            with open('./data/model/mbti_pickle_' + str(i) + '.dat', 'wb') as f:
                pickle.dump(bst, f)
                                             
            # SHOW CONFUSION MATRIX
            if confusion == True:
                        
                # COMPUTE CONFUSION MATRIX
                cnf_matrix = confusion_matrix(y_test, preds)
                np.set_printoptions(precision = 2)
                        
                # PLOT CONFUSION MATRIX
                plt.figure(fig_i)
                fig_i += 1
                self.__plotConfusionMatrix(cm = cnf_matrix, classes = labels)
                
        
    def guessLabel(self, sentence, reddit):
        if reddit == False:
            model = pickle.load(open('./data/model/mbti_kaggle_pickle_5.dat', 'rb'))
            vect = pickle.load(open('./data/model/count_vectorizer_kaggle_pickle_5.dat', 'rb'))
        else:
            model = pickle.load(open('./data/model/mbti_pickle_4.dat', 'rb'))
            vect = pickle.load(open('./data/model/count_vectorizer_pickle_4.dat', 'rb'))
        
        labels = np.genfromtxt('./data/model/labels.csv', dtype = str)
        
        sentence_arr = np.array(sentence)
        X_sentence = vect.transform(sentence_arr.ravel())
        sentence_xgb = xgb.DMatrix(X_sentence)
        
        pred = model.predict(sentence_xgb)
        
        result_sort = np.argsort(pred)
        
        index_1 = result_sort[0, -1]
        index_2 = result_sort[0, -2]
        
        mbti_type_1 = labels[index_1]
        mbti_type_2 = labels[index_2]
        
        return mbti_type_1, mbti_type_2
        