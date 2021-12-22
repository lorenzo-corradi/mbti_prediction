import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
# import nltk # used to locally download stopwords and wordnet, uncomment if necessary
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

class Preprocessing:
    
    df = {}
    
    # LEMMATIZING: REDUCING WORD TO ITS BASIC FORM, RESULTING IN AN ACTUAL LANGUAGE WORD
    __lemmatizer = WordNetLemmatizer()
    # nltk.download('stopwords') # used to locally download stopwords, uncomment if necessary
    # nltk.download('wordnet') # used to locally download wordnet, uncomment if necessary
    __stopwords = stopwords.words("english")
    
    def __init__(self, df):
        self.df = df
    
    def __labelEncoder(self):
        # EXTRACT LABELS
        labels = sorted(self.df['type'].unique().tolist())
        label_encoder = LabelEncoder().fit(labels) # creates numerical labels
        return label_encoder

    def preprocessing(self, remove_stop_words = True):
        
        list_type = []
        list_posts = []
        len_df = len(self.df)
        
        i = 0
        
        for row in self.df.iterrows():
            i += 1
            
            if i % 100 == 0:
                print("%s of %s rows processed" % (i, len_df))

            posts = row[1].posts # select row
            temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'link', posts) # replace urls with word 'link'
            temp = re.sub("[^a-zA-Z]", " ", temp) # keep only words
            temp = re.sub(' +', ' ', temp).lower() # remove spaces > 1
            
            if remove_stop_words:
                temp = " ".join([self.__lemmatizer.lemmatize(w) for w in temp.split(' ') if w not in self.__stopwords]) # divide posts in words and remove stopwords
            else:
                temp = " ".join([self.__lemmatizer.lemmatize(w) for w in temp.split(' ')]) # divide posts in words
                
            result = temp.lstrip() # removing leading whitespace
            type_labelized = self.__labelEncoder().transform([row[1].type])[0]
            list_type.append(type_labelized)
            list_posts.append(result)

        list_posts = np.array(list_posts)
        list_type = np.array(list_type)
        
        return list_posts, list_type, self.df['type']