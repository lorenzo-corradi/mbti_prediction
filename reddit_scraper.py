import praw
import pandas as pd

class RedditScraper(object):
    
    
    dict = { # create dictionary to store data
        "id" : [],
        "author" : [],
        "author_flair_text" : [],
        "body" : [],
        "score" : [],
        "subreddit" : [],
        "created_utc" : [],
        "link_id" : [],
        "parent_id" : [],
        "title" : [],
        "upvote_ratio" : [],
        "post_hint" : []
    }
    
    
    def __init__(self):
        
        # CONNECT TO REDDIT API
        self.reddit = praw.Reddit(client_id = self.client_id,
                            client_secret = self.client_secret,
                            username = self.username,
                            password = self.password,
                            user_agent = self.user_agent)
        
            
    def retrieve_posts(self, post_limit, subreddit_name = "mbti"):
        # INITIALIZATION
        # self.post_limit = post_limit # number of posts to retrieve
        # self.subreddit_name = subreddit_name # change to change subreddit to connect to
        
        # CONNECT TO SPECIFIC SUBREDDIT AND RETRIEVE AN ARBITRARY NUMBER OF POSTS
        self.subreddit = self.reddit.subreddit(subreddit_name)
        top_subreddit = self.subreddit.top(limit = post_limit)
                
        counter = 0
        
        # RETRIEVE POSTS
        for submission in top_subreddit:
            if (submission.author_flair_text != "") and (not submission.is_video) and (not submission.over_18):
                self.dict["id"].append(submission.id)
                self.dict["author"].append(submission.author)
                self.dict["author_flair_text"].append(submission.author_flair_text)
                self.dict["title"].append(submission.title)
                self.dict["score"].append(submission.score)
                self.dict["upvote_ratio"].append(submission.upvote_ratio)
                self.dict["subreddit"].append(submission.subreddit)
                self.dict["created_utc"].append(submission.created_utc)
                self.dict["post_hint"].append(submission.post_hint)
                
                counter += 1
            
                if (counter % 100 == 0):
                    print("iteration #{}".format(counter))
                   
        return self.dict
    
    
    def retrieve_comments_from_submission(self, submissions_id, subreddit_name = "mbti"):
        self.subreddit = self.reddit.subreddit(subreddit_name)
        
        counter = 0
        
        for submission_id in submissions_id:
            # RETRIEVE ALL COMMENTS FOR EACH POST        
            submissions = self.reddit.submission(submission_id)
            
            submissions.comments.replace_more()
            comments = submissions.comments.list()
                
            for comment in comments:
                if comment.author_flair_text:
                    self.dict["id"].append(comment.id)
                    self.dict["author"].append(comment.author)
                    self.dict["author_flair_text"].append(comment.author_flair_text)
                    self.dict["body"].append(comment.body)
                    self.dict["score"].append(comment.score)
                    self.dict["subreddit"].append(comment.subreddit)
                    self.dict["created_utc"].append(comment.created_utc)
                    self.dict["link_id"].append(comment.link_id)
                    self.dict["parent_id"].append(comment.parent_id)
            
            counter += 1
            
            if (counter % 100 == 0):
                print("iteration #{}".format(counter))
            
        return self.dict
    
    
    # CONNECT TO SPECIFIC USERNAME AND RETRIEVE AN ARBITRARY NUMBER OF COMMENTS (IT DOESN'T WORK, ERROR 404)
    def retrieve_comments_from_user(self, subreddit_name = "mbti"):
        
        if (bool(self.dict.get('id'))):
            print("Retrieving users in dictionary...") 
            redditors_id = set(self.dict.get('id'))
            redditors_id = list(redditors_id)
        else:
            # ACCEPTS ONLY ONE USERNAME EACH REQUEST
            redditor_id = str(input("Specify a Reddit username first: "))
            redditors_id = list()
            redditors_id.append(redditor_id)
        
        
        for redditor_id in redditors_id:
            submissions = self.reddit.submission('r/u_' + redditor_id)
            submissions.comments.replace_more()
            comments = submissions.comments.new(limit = None).list()
            
            for comment in comments:
                self.dict["id"].append(comment.id)
                self.dict["author"].append(comment.author)
                self.dict["author_flair_text"].append("")
                self.dict["body"].append(comment.body)
                self.dict["score"].append(comment.score)
                self.dict["subreddit"].append(comment.subreddit)
                self.dict["created_utc"].append(comment.created_utc)
                self.dict["link_id"].append(comment.link_id)
                self.dict["parent_id"].append(comment.parent_id)
                
        return self.dict
    
    def update_flair(self):
        #TODO
        return
