import pandas as pd
from reddit_scraper import RedditScraper

class DataLoader():
    
    df = {}
    
    def __init__(self, post_limit):
        self.post_limit = post_limit
        
                    
class RedditDataLoader(DataLoader):
        
    def __init__(self, post_limit = 0):
        super().__init__(post_limit)
        reddit_scraper = RedditScraper(self.post_limit)
        reddit_scraper.SubredditConnection()
        reddit_scraper.ScrapeData()
        reddit_scraper.CreateDataFrame()
        self.df = pd.DataFrame(reddit_scraper.df)
        
    def getPreBuiltDataFrame(self):
        tmp = pd.read_csv('data/reddit/reddit_mbti_data.csv')
        self.df = pd.DataFrame(tmp)
        return self.df
    
    def getDataFrame(self):
        return self.df
    
    def addFeaturesForVisualization(self):
        # ADD FEATURES TO DATAFRAME
        self.df['words'] = self.df['posts'].apply(lambda x: len(x.split()))
        self.df['http'] = self.df['posts'].apply(lambda x: x.count('http'))
        self.df['question'] = self.df['posts'].apply(lambda x: x.count('?'))
        self.df['exclamation'] = self.df['posts'].apply(lambda x: x.count('!'))
        self.df['ellipsis'] = self.df['posts'].apply(lambda x: x.count('..')) # many people all over the world use only two dots to indicate ellipsis
               
        return self.df
        
        
class KaggleDataLoader(DataLoader):
    
    def __init__(self):
        tmp = pd.read_csv('data/kaggle/kaggle_mbti_data.csv')
        self.df = pd.DataFrame(tmp)
    
    def getDataFrame(self):
        return self.df
    
    def addFeaturesForVisualization(self):
        # ADD FEATURES TO DATAFRAME
        self.df['words'] = self.df['posts'].apply(lambda x: len(x.split()) / 50)
        self.df['http'] = self.df['posts'].apply(lambda x: x.count('http') / 50)
        self.df['question'] = self.df['posts'].apply(lambda x: x.count('?') / 50)
        self.df['exclamation'] = self.df['posts'].apply(lambda x: x.count('!') / 50)
        self.df['ellipsis'] = self.df['posts'].apply(lambda x: x.count('..') / 50) # many people all over the world use only two dots to indicate ellipsis
                
        return self.df