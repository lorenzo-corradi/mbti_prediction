import pandas as pd


class DataLoader():
    
    def load_data_pandas(self):
        X = pd.read_csv('C:/Users/LCorradi/Desktop/University/Passed Exams and Python Envs/Python Envs of Passed Exams/data/mbti.csv', index_col = False, dtype = {"title": str, "post_hint": str}) 
        # pandas was unsure about dtype for columns 9 and 11
        return X
    
    
    def load_clean_data_pandas(self):
        X = pd.read_csv('C:/Users/LCorradi/Desktop/University/Passed Exams and Python Envs/Python Envs of Passed Exams/data/mbti_clean.csv') 
        # pandas was unsure about dtype for columns 9 and 11
        return X
    
    def load_preprocessed_data_pandas(self):
        X = pd.read_pickle('C:/Users/LCorradi/Desktop/University/Passed Exams and Python Envs/Python Envs of Passed Exams/data/mbti_preprocess.pkl')
        return X
    
        
    def load_aggregated_data_pandas(self):
        X = pd.read_pickle('C:/Users/LCorradi/Desktop/University/Passed Exams and Python Envs/Python Envs of Passed Exams/data/mbti_aggregated.pkl')
        return X
    
    
    def load_filtered_data_pandas(self):
        X = pd.read_pickle('C:/Users/LCorradi/Desktop/University/Passed Exams and Python Envs/Python Envs of Passed Exams/data/mbti_filtered.pkl')
        return X
    
        
    def load_unnested_data_pandas(self):
        X = pd.read_pickle('C:/Users/LCorradi/Desktop/University/Passed Exams and Python Envs/Python Envs of Passed Exams/data/mbti_unnested.pkl')
        return X
    
    
    def path_pretrained_word2vec(self):
        path_pretrained_word2vec = "C:/Users/LCorradi/Desktop/University/Passed Exams and Python Envs/Python Envs of Passed Exams/data/GoogleNews-vectors-negative300.bin"
        return path_pretrained_word2vec
    
    
    def load_vectorized_data_pandas(self):
        X = pd.read_pickle('C:/Users/LCorradi/Desktop/University/Passed Exams and Python Envs/Python Envs of Passed Exams/data/mbti_vectorized.pkl.gz', compression = 'gzip')
        return X

    
        