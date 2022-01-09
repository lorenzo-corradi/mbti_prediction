import pandas as pd
import re
from data_loader import DataLoader
import time

class Cleaning:
    
    
    def flair_list(self, X):
        return [str(i) for i in list(set(X))]
    
    
    def merge_posts_and_comments(self, X_posts, X_comments):
        # find better way to write this
        X_comments = X_comments.fillna(X_posts)
        
        return X_comments, None
    
    
    def remove_urls(self, X, string = "link"):
        # remove urls, replace with word 'link'
        string = string.center(len(string) + 2) # one space padding both on left and right
        X = X.str.replace(r'http\S+|www\S+', string, regex = True)
        
        return X
    
    
    def remove_emoticons(self, X, string = "emoji"):
        # remove emoticons, replace with word 'emoji'
        string = string.center(len(string) + 2) # one space padding both on left and right
        X = X.str.replace(r'[^\w\s#@/:%.,_-]', string, flags = re.UNICODE, regex = True)
        
        return X
    
    
    def remove_consecutive_spaces(self, X):
        # remove consecutive spaces
        X = X.str.replace(r' +', ' ', regex = True)
        
        return X
    
    
    def convert_digits_to_words(self):
        # TODO
        return
    
    
    def remove_flair_occurrences(self, X, y, string = "flair"):
        
        unique_flair_list = self.flair_list(y)
        
        string = string.center(len(string) + 2) # one space padding both on left and right
        replacements = {}
        
        # set up dictionary with labels as keys and string as value
        for flair in unique_flair_list:
            replacements[flair] = string
        
        # deal with escape characters such as symbols
        replacements = dict((re.escape(k), v) for k, v in replacements.items())
        
        # set up regex expression to search for all labels in text
        pattern = re.compile("|".join(replacements.keys()))
        
        temp = list()
        
        # apply substitutions (O(n) for now)
        for corpus in X:
            temp.append(pattern.sub(lambda f: replacements[re.escape(f.group())], corpus))
            
        X = pd.Series(temp)
        
        return X
    
    
    # BEWARE: HARDCODED SOLUTION
    def _convert_noisy_labels(self, y):
        y.loc[(y == 'bot') | (y == 'mr. roboto')] = None
        
        return y
    
    
    def _get_unlabeled_data(self, X, y):
        X_nolabel = X[y.isnull()]
        
        return X_nolabel

    
    def apply_labels(self, X, y):
        
        y = self._convert_noisy_labels(y)
        X_nolabel = self._get_unlabeled_data(X, y)
        
        unique_flair_list = self.flair_list(y)
        
        for flair in unique_flair_list: # take each flair
            for idx, corpus in enumerate(X_nolabel): # take each unlabeled observation
                if flair in corpus: # if flair is in the text of unlabeled observation
                    # TODO: what if an observation has more than one flair in the text?
                    index = X_nolabel.index[idx] # find unlabeled observation index
                    y[index] = flair # set label on target variable on that specific index
        
        return y
    
    
    def remove_unlabeled_data(self, X, y):
        y = self._convert_noisy_labels(y)
        
        X = X.loc[pd.notnull(y)]
        X.reset_index(drop = True, inplace = True)
        
        return X
    
    
    def remove_unuseful_data(self, X, threshold):
        # TODO: drop row if body has less than arbitrary number of characters
        return
    
    
    def remove_unuseful_columns(self, X, threshold = 0.75):

        X.dropna(axis = 1, thresh = int(threshold * len(X)), inplace = True)
        return X
    
    
    def apply_lowercase(self, X):
        X = X.applymap(lambda f: str(f).lower() if type(f) == str else f)
        
        return X
    
    
    def export_data(self, X):
        X.to_csv('./data/mbti_clean.csv', index = False)
        
        return
        
    
if __name__ == "__main__":
    
    data_loader = DataLoader()
    X = data_loader.load_data_pandas()
    
    start_time = time.time()
    print("--- %s seconds ---" % (round(time.time() - start_time, 2)))
    
    cleaning = Cleaning()
    X = cleaning.apply_lowercase(X)
    
    X['body'], X['title'] = cleaning.merge_posts_and_comments(X['title'], X['body'])
    
    X['body'] = cleaning.remove_urls(X['body'])
    
    X['body'] = cleaning.remove_emoticons(X['body'])
    
    X['body'] = cleaning.remove_flair_occurrences(X['body'], X['author_flair_text'])
    
    X['body'] = cleaning.remove_consecutive_spaces(X['body'])
    
    X['author_flair_text'] = cleaning.apply_labels(X['body'], X['author_flair_text'])
    
    X = cleaning.remove_unlabeled_data(X, X['author_flair_text'])
    
    X = cleaning.remove_unuseful_columns(X)
    
    cleaning.export_data(X)
    
    print("--- %s seconds ---" % (round(time.time() - start_time, 2))) # time elapsed: 18 seconds for 300k row pandas dataframe