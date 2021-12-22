import praw
import pandas as pd

class RedditScraper:
    
    __client_id = 'B1DsaKlNLNzd7Q'
    __client_secret = 'TyViPR-RF2t0xZiZqgfqfrkaxF-GTg'
    __username = 'univr_prog_exam'
    __password = 'univr_prog_exam'
    __user_agent = 'mbti_prog_exam'
    
    subreddit_name = 'mbti' # change to change subreddit to connect to
    
    data_dict = { # create dictionary for posts and comments
        "type" : [],
        "posts" : []
    }
    
    df = {}
    
    def __init__(self, post_limit):
        self.post_limit = post_limit # number of posts to retrieve
    
    def __RedditAPIConnection(self):
        # CONNECT TO REDDIT API
        reddit = praw.Reddit(client_id = self.__client_id,
                            client_secret = self.__client_secret,
                            # username = self.__username,
                            # password = self.__password,
                            user_agent = self.__user_agent)
        return reddit
        
    def SubredditConnection(self):
        reddit = self.__RedditAPIConnection()
        
        # CONNECT TO SPECIFIC SUBREDDIT (r/mbti) AND RETRIEVE AN ARBITRARY NUMBER OF POSTS
        subreddit = reddit.subreddit(self.subreddit_name)
        top_mbti = subreddit.top(limit = self.post_limit)
        return top_mbti
            
    def ScrapeData(self):
        # INITIALIZATION
        top_mbti = self.SubredditConnection()
                
        # RETRIEVE POSTS
        for submission in top_mbti:
            if submission.author_flair_text:
                self.data_dict["type"].append(submission.author_flair_text)
                self.data_dict["posts"].append(submission.title)

            # RETRIEVE ALL COMMENTS FOR EACH POST
            submission.comments.replace_more()
            comments = submission.comments.list()
            for comment in comments:
                if comment.author_flair_text:
                    self.data_dict["type"].append(comment.author_flair_text)
                    self.data_dict["posts"].append(comment.body)
                
    def CreateDataFrame(self):
        # PUT DICTIONARY INTO DATAFRAME AND CREATE CSV
        self.df = pd.DataFrame(self.data_dict)
        # self.df.to_csv('./data/reddit/reddit_mbti_data.csv', index = False)
        return self.df