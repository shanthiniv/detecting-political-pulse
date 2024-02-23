import tweepy
import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import statistics
from sklearn.neighbors import KNeighborsClassifier

with open("twitter_credentials.json", "r") as f:
    credentials = json.load(f)

auth = tweepy.OAuthHandler(credentials["consumer_key"], credentials["consumer_secret"])
auth.set_access_token(credentials["access_token"], credentials["access_token_secret"])
api = tweepy.API(auth, wait_on_rate_limit=True)
def collect_data(politician_handle):
def process_tweets(political_social_media.csv):
def plot_polarity_subjectivity_scatter():
def plot_knn_boundary():
def generate_wordcloud():
def plot_engagement_vs_polarity():
def plot_average_tweets():

if __name__ == "__main__":
    politician_handles = ["treyradel", "McConnellPress", "RepSchrader"]
    for handle in politician_handles:
        collect_data(handle)
        process_tweets("tweets_" + handle + ".csv")
        plot_polarity_subjectivity_scatter()
        plot_knn_boundary()
        generate_wordcloud()
        plot_engagement_vs_polarity()
        plot_average_tweets()
