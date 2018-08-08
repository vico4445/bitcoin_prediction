# import tweepy library for twitter api access and textblob libary for sentiment analysis
import csv
import tweepy
import numpy as np
from textblob import TextBlob
import datetime
import time

"""
    Ce script ouvre un strem à twitter. Pour chaque tweet récupéré, un sentiment est affecté ainsi qu'une polarity (>0 sentiment positif, <0 sentiment négatif)
    Il enregistre les tweet dans le fichier 'live_tweet.csv' (argument du script)
"""

def main():

    # set twitter api credentials
    consumer_key= '5lOwfiKQL6f3GU3HeFJDHfL8p'
    consumer_secret= 'OdvoN1brET3p5tDko0w2tIg57C1gny3VMJgKP8RWbIualLKuhv'
    access_token='1026568968079851520-bJ1k7kJswYl0eD6tW2bj4ZpFgNPTJC'
    access_token_secret='OI7sGtz3NiOe0F9Dpj7T9Gt27qs6qFuw9r0mv9keevSnE'

    # set path of csv file to save sentiment stats
    path = 'live_tweet.csv'
    f = open(path,"a")
    f1 = open('tweet_data','a', encoding='utf-8')
    # access twitter api via tweepy methods
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    twitter_api = tweepy.API(auth)

    while True:
        # fetch tweets by keywords
        tweets = twitter_api.search(q=['bitcoin, price, crypto'], count=100)

        # get polarity
        polarity = get_polarity(tweets,f1)
        sentiment = np.mean(polarity)

        # save sentiment data to csv file
        f.write(str(sentiment))
        f.write(","+datetime.datetime.now().strftime("%y-%m-%d-%H-%M"))
        f.write("\n")
        f.flush()
        time.sleep(60)
    

def get_polarity(tweets,f):
    # run polarity analysis on tweets

    tweet_polarity = []

    for tweet in tweets:
        print(tweet.text)
        f.write(tweet.text+'\n')
        analysis = TextBlob(tweet.text)
        tweet_polarity.append(analysis.sentiment.polarity)

    return tweet_polarity

main()
