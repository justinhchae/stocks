import tweepy
from decouple import config

class GetTwitter():
    def __init__(self):
        self.api_key = config('TWITTER_API_KEY')
        self.api_secret = config('TWITTER_API_SECRET')
        self.token = config('TWITTER_TOKEN')

        self.auth = tweepy.AppAuthHandler(self.api_key, self.api_secret)
        self.api = tweepy.API(self.auth, wait_on_rate_limit=True)

    def test_func(self):
        for tweet in tweepy.Cursor(self.api.search
                , q=['#gme', '#gamestop', 'gme', 'game stop']
                , lang="en"
                , result_type="mixed"

                                   ).items(5):
            print(tweet.text)



GetTwitter().test_func()