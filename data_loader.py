import pandas as pd
# import csv
# import numpy as np

class DataLoader():
        
    def __init__(self):
        return
    
    # take a look at new_reddit_mbti: it has unlabeled data!
    
    # BETTER TO USE PANDAS, THIS FUNCTION NEEDS TWEAKING
    # def load_data_numpy(self):
        
    #     # ADD CHECK IF DICT IS ALREADY FULL
        
    #     X = RedditScraper().dict
    #     X.pop("author_flair_text", None)
    #     y = list()
        
    #     with open('./data/mbti.csv', mode = 'r', newline = '', encoding = 'utf-8') as file:
            
    #         reader = csv.reader(file)
    #         column_names = list(next(reader))
            
    #         # TODO: FIND A WAY TO GET RID OF THIS
    #         for i, row in enumerate(reader):
                
    #             if (i < 40000):
                
    #                 X["id"].append(row[column_names.index("id")])
    #                 X["author"].append(row[column_names.index("author")])
    #                 y.append(row[column_names.index("author_flair_text")])
    #                 X["body"].append(row[column_names.index("body")])
    #                 X["score"].append(row[column_names.index("score")])
    #                 X["subreddit"].append(row[column_names.index("subreddit")])
    #                 X["created_utc"].append(row[column_names.index("created_utc")])
    #                 X["link_id"].append(row[column_names.index("link_id")])
    #                 X["parent_id"].append(row[column_names.index("parent_id")])
    #                 X["title"].append(row[column_names.index("title")])
    #                 X["upvote_ratio"].append(row[column_names.index("upvote_ratio")])
    #                 X["post_hint"].append(row[column_names.index("post_hint")])
                
    #     # FIND A WAY TO GET MORE DATA (ONLY 40000 ROWS) AND BETTER DATA TYPES (EVERYTHING IS AN OBJECT)
    #     # FIND A QUICK WAY TO GET FORMATS FROM ROWS
    #     # formats = ['str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str', 'str']
    #     # dtype = dict(names = column_names, formats = formats)
        
    #     X = np.array(list(X.values())).T
        
    #     return X, y
                           
    
    def load_data_pandas(self):
        X = pd.read_csv('./data/mbti.csv', index_col = False, dtype = {"title": str, "post_hint": str}) 
        # pandas was unsure about dtype for columns 9 and 11
        return X
    
    def load_clean_data_pandas(self):
        X = pd.read_csv('./data/mbti_clean.csv', index_col = False) 
        # pandas was unsure about dtype for columns 9 and 11
        return X
    
        