from data_loader import DataLoader
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# objective: remove stopwords and punctuation
# custom bi-grams and tri-grams without stopwords removal
# idea: create different .csv: one with stopwords, one without stopwords, one with comments of >= x char, one with less

class Preprocessing:
    
    
    def init_spacy(self, X):
        
        nlp = spacy.load("en_core_web_sm")
        docs = nlp.pipe(X.astype('unicode').values)
        
        return docs
    
    
    def remove_punctuation(self, docs, X):
        
        X_nopunct = [[w.lemma_ for w in doc if not w.is_punct] for doc in docs]
                
        X_nopunct = pd.Series(X_nopunct)        
        
        return X_nopunct
    
    
    def remove_stopwords(self, docs, X):
        
        X_nostop = [[w.lemma_ for w in doc if not w.is_stop] for doc in docs]
        
        X_nostop = pd.Series(X_nostop)
        
        return X_nostop
    
    
    def remove_numbers(self, docs, X):
        
        X_nonums = [[w.lemma_ for w in doc if not w.like_num] for doc in docs]

        X_nonums = pd.Series(X_nonums)
        
        return X_nonums
    
    
    def find_bigrams(self, X):
        
        # bug due to data type
        X_bigrams = [[b for b in zip(l.split(' ')[:-(len(X) - 1)], l.split(' ')[(len(X) - 1):])] for l in X.values]
        
        X_bigrams = pd.Series(X_bigrams)
        
        return X_bigrams
    
        
    def export_data(self, X, string):
        
        X.to_csv('./data/mbti_clean_'+ string + '.csv', index = False)
        
        return
    
    
    # def __labelEncoder(self):
    #     # EXTRACT LABELS
    #     labels = sorted(self.df['type'].unique().tolist())
    #     label_encoder = LabelEncoder().fit(labels) # creates numerical labels
    #     return label_encoder
    
if __name__ == "__main__":
    
    data_loader = DataLoader()
    X = data_loader.load_clean_data_pandas()
    
    preprocessing = Preprocessing()
    