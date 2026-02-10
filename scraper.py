import pandas as pd
import requests
from bs4 import BeautifulSoup
import praw
from praw.models import MoreComments
import time
import sys

SUBREDDIT_URL = "https://www.reddit.com/r/shittysuperpowers/"

def forum_link_generator(SUBREDDIT_URL):
    # mimic browser visit 
    headers = {'User-Agent': 'Mozilla/5.0'}
    page = requests.get(SUBREDDIT_URL, headers=headers)
    soup = BeautifulSoup(page.text, "html.parser")

def forum_scraper(url):
    reddit = praw.Reddit(user_agent="USER_AGENT", 
                         client_id="CLIENT_ID", client_secret="CLIENT_SECRET")
    posts = []
    for submission in reddit.subreddit("shittysuperpowers").top(limit=10, time_filter="day"):
        for top_level_comment in submission.comments:
            if isinstance(top_level_comment, MoreComments):
                continue
    posts.append(top_level_comment.body)
    posts = pd.DataFrame(posts,columns=["body"])
    indexNames = posts[(posts.body == '[removed]') | (posts.body == '[deleted]')].index
    posts.drop(indexNames, inplace=True)
    print(posts)

def main():
    forum_scraper(SUBREDDIT_URL) 

if __name__ == "__main__":
    main()