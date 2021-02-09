import praw
import pandas as pd
import datetime as dt
from decouple import config

class GetReddit():
    def __init__(self):
        self.client_id = config('CLIENT_ID')
        self.client_secret = config('CLIENT_SECRET')
        self.user_agent = config('USER_AGENT')
        self.username = config('USER_NAME')
        self.password = config('USER_PASSWORD')

        self.reddit = praw.Reddit(client_id=self.client_id
                                  , client_secret=self.client_secret
                                  , user_agent=self.user_agent
                                  , username=self.username
                                  , password=self.password)

        self.wallstreet = "wallstreetbets"

        self.reddit_dict = {"title":[]
                            ,"score":[]
                            ,"id": []
                            ,"url": []
                            ,"comms_num":[]
                            ,"created":[]
                            ,"body":[]}

        def set_subreddit(topic=None):
            if topic is None:
                topic = self.wallstreet
            return self.reddit.subreddit(topic)

        self.subreddit = set_subreddit()

    def get_date(self, x):
        return dt.datetime.fromtimestamp(x)

    def get_threads(self, thread_type="top", limit=None):
        if thread_type == "top":
            self.top_subreddit = self.subreddit.top(limit=limit)

            for i in self.top_subreddit:
                self.reddit_dict["title"].append(i.title)
                self.reddit_dict["score"].append(i.score)
                self.reddit_dict["id"].append(i.id)
                self.reddit_dict["url"].append(i.url)
                self.reddit_dict["comms_num"].append(i.num_comments)
                self.reddit_dict["created"].append(i.created)
                self.reddit_dict["body"].append(i.selftext)

            df = pd.DataFrame(self.reddit_dict)
            df["created"] = df["created"].apply(self.get_date)

            print(len(df))

            df.to_csv('data/wsb_tops.csv', index=False)
            df.to_pickle('data/wsb_tops.pickle', protocol=2)

    #TODO: create search/crawler for posts on gamestop in three periods
    # period 1: 1 DEC 2019 to 1 DEC 2020
    # period 2: 1 DEC 2020 to 30 JAN 2021
    # period 3: 30 JAN 2021 to present day
    # https://abcnews.go.com/Business/gamestop-timeline-closer-saga-upended-wall-street/story?id=75617315



