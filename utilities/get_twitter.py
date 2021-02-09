import twitter
from decouple import config

class GetReddit():
    def __init__(self):
        self.api_key = config('TWITTER_API_KEY')
        self.api_secret = config('TWITTER_API_SECRET')
        self.token = config('TWITTER_TOKEN')

        self.t = twitter.Api(access_token_key=self.api_key, 
                             access_token_secret=self.api_secret)