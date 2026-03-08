import pandas as pd
import requests
from requests.exceptions import RequestException
from bs4 import BeautifulSoup
# import praw
# from praw.models import MoreComments
import json
import time
import sys
import os
from dotenv import load_dotenv

load_dotenv()

# only for Bright Data
API_TOKEN = os.getenv("API_TOKEN")
ZONE = os.getenv("ZONE")

SUBREDDIT_URL = "https://www.reddit.com/r/finance/top.json?limit=1000"
SUBREDDIT_URL_OLD = "https://old.reddit.com/r/shittysuperpowers/new/"

OUTPUT_FILE = "posts.json"
STATE_FILE = "state.json" # keeps track of seen post ids

# seen posts
if os.path.exists(STATE_FILE) and os.path.getsize(STATE_FILE) > 0:
    with open(STATE_FILE, "r") as f:
        seen_ids = set(json.load(f))
else:
    seen_ids = set()

# only usable for old reddit
def scrape_subreddit():

    headers = {
        "User-Agent": "script:reddit-cluster: (by u/InternalReference258)"
    }

    new_posts = []
    url = SUBREDDIT_URL_OLD

    while url:
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except RequestException as e:
            print(f"Request failed: {e}")
            break

        soup = BeautifulSoup(response.text, "html.parser")

        posts = soup.find_all("div", class_="thing")

        stop_pagination = False

        for post in posts:
            post_id = post.get("data-fullname")

            if not post_id:
                continue

            if post_id in seen_ids:
                stop_pagination = True
                break

            title_tag = post.find("a", class_="title")
            if not title_tag:
                continue

            title = title_tag.text.strip()
            post_link = "https://old.reddit.com" + title_tag["href"]

            new_posts.append({
                "id": post_id,
                "title": title,
                "url": post_link
                # "body": body_text,
                # "image": image_url
            })
            seen_ids.add(post_id)
            # time.sleep(1)

        if stop_pagination:
            break

        # go to next page
        next_button = soup.find("span", class_="next-button")
        if next_button:
            url = next_button.find("a")["href"]
            time.sleep(1)
        else:
            break

    # save new posts
    if new_posts:
        # add to posts.json
        if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0:
            with open(OUTPUT_FILE, "r") as f:
                existing = json.load(f)
        else:
            existing = []
        existing.extend(new_posts)
        with open(OUTPUT_FILE, "w") as f:
            json.dump(existing, f, indent=2)

        # update state.json with seen ids
        with open(STATE_FILE, "w") as f:
            json.dump(list(seen_ids), f)

    print(f"Collected {len(new_posts)} new posts from {url}.")

# next step after scrape_subreddit(). Adds body text and images to each post
def scrape_posts():
    headers = {
        "User-Agent": "script:reddit-cluster: (by u/InternalReference258)"
    }
    
    with open(OUTPUT_FILE, "r") as f:
        posts = json.load(f)
    
    update_cnt = 0
    scraped_cnt = 0
    skip_cnt = 0
    total_cnt = 0
    for post in posts:
        total_cnt += 1
        # skip scraped posts
        if post.get("scraped"):
            scraped_cnt += 1
            continue

        url = post["url"]
        response = requests.get(url=url, headers=headers, timeout=10)
        if response.status_code != 200:
            # print(f"Skipping {url} — Status {response.status_code}")
            skip_cnt += 1
            continue
        
        soup = BeautifulSoup(response.text, "html.parser")

        # body text
        thing = soup.find("div", id=lambda x: x and x.startswith("thing_t3_"))
        body_div = thing.find("div", class_="usertext-body")
        body_text = body_div.get_text(separator="\n").strip() if body_div else ""

        # image
        og_image = soup.find("meta", property="og:image")
        image_url = og_image["content"] if og_image else ""
        
        post["body"] = body_text
        post["image_url"] = image_url
        post["scraped"] = True

        update_cnt += 1

        # current testing shows sleeping may not be necessary
        # time.sleep(2)

    print(f"Skipped: {skip_cnt} // Scraped: {scraped_cnt + update_cnt} // Total: {total_cnt}")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(posts, f, indent=2)

def main():
    start = time.time()
    scrape_subreddit()
    print(f"Execution time: {time.time() - start:2f} seconds")

    start = time.time()
    scrape_posts()
    print(f"Execution time: {time.time() - start:2f} seconds")
if __name__ == "__main__":
    main()