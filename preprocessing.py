from data_loader import DataLoader
import pandas as pd
from gensim.parsing.preprocessing import STOPWORDS
from gensim.parsing import preprocessing
import gensim.utils as utils
import time
import pickle


class Preprocessing:


    def remove_non_alphanumeric(self, X):

        start_time = time.time()

        X_nonum = [preprocessing.strip_non_alphanum(str(sentence)) for sentence in X]

        print("--- Non alphanumeric characters removed. Time elapsed: %s seconds ---" % (round(time.time() - start_time, 2)))

        return X_nonum


    def remove_stopwords_proj(self, X):

        start_time = time.time()

        X_num_words = [len(str(sentence).split()) for sentence in X.tolist()]

        X_nostop = [preprocessing.remove_stopwords(str(sentence), STOPWORDS) for sentence in X]

        X_num_stopwords = [len(str(sentence).split()) for sentence in X_nostop]

        X_num_stopwords = [X_num_words - X_num_stopwords for X_num_words, X_num_stopwords in zip(X_num_words, X_num_stopwords)]

        print("--- Stopwords removed. Time elapsed: %s seconds ---" % (round(time.time() - start_time, 2)))

        return X_nostop, X_num_stopwords


    def remove_short_words(self, X):

        start_time = time.time()

        X_num_words = [len(str(sentence).split()) for sentence in X.tolist()]

        X_noshort = [preprocessing.strip_short(str(sentence), minsize = 4) for sentence in X]

        X_num_short = [len(str(sentence).split()) for sentence in X_noshort]

        X_num_short = [X_num_words - X_num_short for X_num_words, X_num_short in zip(X_num_words, X_num_short)]

        print("--- Short words removed. Time elapsed: %s seconds ---" % (round(time.time() - start_time, 2)))

        return X_noshort, X_num_short


    def remove_punctuation(self, X):

        start_time = time.time()

        # TODO: write loop in LIST COMPREHENSION
        X_nopunct = []

        for sentence in X:
            sentence = str(sentence).replace('\n', ' ').replace('\r', '')
            X_nopunct.append(preprocessing.strip_punctuation(str(sentence)))

        print("--- Punctuation removed. Time elapsed: %s seconds ---" % (round(time.time() - start_time, 2)))

        return X_nopunct


    def tokenize_proj(self, X):

        start_time = time.time()

        X_token = [list(utils.tokenize(str(sentence), deacc = True, lowercase = True)) for sentence in X]

        print("--- Tokenized. Time elapsed: %s seconds ---" % (round(time.time() - start_time, 2)))

        return X_token


    def aggregate_by_author(self, X):

        X_aggregated = pd.DataFrame(X[['author_flair_text', 'tokens', 'score', 'num_short', 'num_stopwords']].groupby(
            by = X['author'], as_index = False).agg(
                tokens = ('tokens', 'sum'), num_short = ('num_short', 'sum'), author_flair_text = ('author_flair_text', 'first'), num_stopwords = ('num_stopwords', 'sum'), score = ('score', 'sum')))

        return X_aggregated


    # BEWARE: HARDCODED FUNCTION THAT CREATES ANOTHER DATASET!
    def remove_users_few_tokens(self, X, threshold = 20):

        start_time = time.time()

        # apply aggregation first

        X_filtered = X[X['tokens'].str.len() >= threshold]
        X_filtered.index = range(len(X_filtered.index))

        print("--- Removed users with few tokens. Time elapsed: %s seconds ---" % (round(time.time() - start_time, 2)))

        return X_filtered


    def train_test_split_proj(self, X, y):

        start_time = time.time()

        X_splitted = X.assign(splitting = 'train')
        X_splitted.loc[y.isna() == True, 'splitting'] = 'test'

        print("--- Train-test separation applied. Time elapsed: %s seconds ---" % (round(time.time() - start_time, 2)))

        return X_splitted
    
    
    def add_token_length(self, X):
        
        start_time = time.time()
        
        X_len_tokens = X.assign(len_tokens = X['tokens'].str.len())
        
        print("--- Added token length. Time elapsed: %s seconds ---" % (round(time.time() - start_time, 2)))
        
        return X_len_tokens
    
    
    def unnest_tokens(self, X):

        start_time = time.time()

        X_unnest = X.explode('tokens')
        X_unnest['body_index'] = X_unnest.index
        X_unnest.index = range(len(X_unnest))

        print("--- Tokens unnested. Time elapsed: %s seconds ---" % (round(time.time() - start_time, 2)))

        return X_unnest
    
    
    # TODO: DOES NOT WORK!
    def split_labels(self, X):
        
        start_time = time.time()
        
        columns = ['e/i', 'n/s', 'f/t', 'j/p']
        
        X_new_labels = pd.DataFrame(columns = columns)
        
        for i in range(len(X['author_flair_text'][:50])):
            # print(list(str(X['author_flair_text'][i])))
            X_new_labels.append(pd.DataFrame([list(str(X['author_flair_text'][i]))]))
        
            
        #X_new_labels = pd.DataFrame([list(X['author_flair_text'][0])], columns = columns)
        print(X_new_labels[:50])
    
        print("--- Labels splitted. Time elapsed: %s seconds ---" % (round(time.time() - start_time, 2)))
        
        return X_new_labels
        

    #TODO: remove full path
    def export_data(self, X):

        start_time = time.time()

        X.to_pickle('C:\\Users\\LCorradi\\Desktop\\University\\Passed Exams and Python Envs\\Python Envs of Passed Exams\\data\\mbti_preprocess.pkl')

        X.to_csv('C:\\Users\\LCorradi\\Desktop\\University\\Passed Exams and Python Envs\\Python Envs of Passed Exams\\data\\mbti_preprocess.csv', index = False)

        print("--- Dataset uploaded in .csv and .pkl.gz. Time elapsed: %s seconds ---" % (round(time.time() - start_time, 2)))

        return


    #TODO: remove full path
    def export_aggregated_data(self, X):

        start_time = time.time()

        X.to_pickle('C:\\Users\\LCorradi\\Desktop\\University\\Passed Exams and Python Envs\\Python Envs of Passed Exams\\data\\mbti_aggregated.pkl')

        X.to_csv('C:\\Users\\LCorradi\\Desktop\\University\\Passed Exams and Python Envs\\Python Envs of Passed Exams\\data\\mbti_aggregated.csv', index = False)

        print("--- Dataset uploaded in .csv and .pkl.gz. Time elapsed: %s seconds ---" % (round(time.time() - start_time, 2)))

        return


    #TODO: remove full path
    def export_filtered_data(self, X):

        start_time = time.time()
        
        X.to_pickle('C:\\Users\\LCorradi\\Desktop\\University\\Passed Exams and Python Envs\\Python Envs of Passed Exams\\data\\mbti_filtered.pkl')

        X.to_csv('C:\\Users\\LCorradi\\Desktop\\University\\Passed Exams and Python Envs\\Python Envs of Passed Exams\\data\\mbti_filtered.csv', index = False)

        print("--- Dataset uploaded in .csv and .pkl.gz. Time elapsed: %s seconds ---" % (round(time.time() - start_time, 2)))

        return


    #TODO: remove full path
    def export_unnested_data(self, X):

        start_time = time.time()

        X.to_pickle('C:\\Users\\LCorradi\\Desktop\\University\\Passed Exams and Python Envs\\Python Envs of Passed Exams\\data\\mbti_unnested.pkl')

        X.to_csv('C:\\Users\\LCorradi\\Desktop\\University\\Passed Exams and Python Envs\\Python Envs of Passed Exams\\data\\mbti_unnested.csv', index = False)

        print("--- Dataset uploaded in .csv and .pkl.gz. Time elapsed: %s seconds ---" % (round(time.time() - start_time, 2)))

        return


if __name__ == "__main__":

    preprocess = Preprocessing()

    data_loader = DataLoader()
    # X = data_loader.load_clean_data_pandas()

    start_time = time.time()

    # X['body_preprocess'] = pd.Series(preprocess.remove_punctuation(X['body']))
    # X['body_preprocess'] = pd.Series(preprocess.remove_non_alphanumeric(X['body']))
    # X['body_preprocess'], X['num_stopwords'] = pd.Series(preprocess.remove_stopwords_proj(X['body']))
    # X['body_preprocess'], X['num_short'] = pd.Series(preprocess.remove_short_words(X['body']))
    # X['tokens'] = pd.Series(preprocess.tokenize_proj(X['body_preprocess']))

    # preprocess.export_data(X)

    # X = data_loader.load_preprocessed_data_pandas()

    # X_aggregated = pd.DataFrame(preprocess.aggregate_by_author(X))
    
    # preprocess.export_aggregated_data(X_aggregated)

    X_aggregated = data_loader.load_aggregated_data_pandas()
    
    X_filtered = preprocess.remove_users_few_tokens(X_aggregated, threshold = 20)
    X_filtered = pd.DataFrame(preprocess.train_test_split_proj(X_filtered, X_filtered['author_flair_text']))
    X_filtered = preprocess.add_token_length(X_filtered)
    
    preprocess.export_filtered_data(X_filtered)
    
    X_filtered = data_loader.load_filtered_data_pandas()
            
    # X_filtered = preprocess.split_labels(X_filtered)
    
    X_unnested = preprocess.unnest_tokens(X_filtered)
        
    preprocess.export_unnested_data(X_unnested)
    

    print("--- Dataset preprocessed and filtered. Time elapsed: %s seconds ---" % (round(time.time() - start_time, 2)))
    # 3 minutes top to process 300k rows