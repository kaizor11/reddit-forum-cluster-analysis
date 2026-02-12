import sys
import time

from scraper import scrape_subreddit, scrape_posts, SUBREDDIT_URL_OLD

scrape_interval_min = sys.argv[1]

def scrape():
    print(f"Scraping from {SUBREDDIT_URL_OLD}...")

    start = time.time()
    scrape_subreddit()
    print(f"Execution time: {time.time() - start:2f} seconds")

    start = time.time()
    scrape_posts()
    print(f"Execution time: {time.time() - start:2f} seconds")

    print(f"Sleeping for {scrape_interval_min} minutes")
    time.sleep(scrape_interval_min * 60)

def store():
    pass

def preprocess():
    pass

def cluster():
    pass