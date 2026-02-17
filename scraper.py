import pandas as pd
import requests
from bs4 import BeautifulSoup
import praw
from praw.models import MoreComments
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
        response = requests.get(url, headers=headers, timeout=10)
        # if response.status_code == 429:
        #     print("Rate limited. Sleeping 30 seconds.")
        #     time.sleep(30)
        #     continue

        response.raise_for_status()

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
    skip_cnt = 0
    for post in posts:
        # skip scraped posts
        if "body" in post and post["body"]:
            continue

        url = post["url"]
        response = requests.get(url=url, headers=headers, timeout=10)
        if response.status_code != 200:
            # print(f"Skipping {url} — Status {response.status_code}")
            skip_cnt += 1
            continue
        soup = BeautifulSoup(response.text, "html.parser")

        # body text
        body_div = soup.find("div", class_="usertext-body")
        body_text = body_div.get_text(separator="\n").strip() if body_div else ""

        # image
        og_image = soup.find("meta", property="og:image")
        image_url = og_image["content"] if og_image else ""

        # only update if there is data
        if body_text or image_url:
            post["body"] = body_text
            post["image_url"] = image_url

            # keep track of body text updates
            update_cnt += 1
        
        # current testing shows sleeping may not be necessary
        # time.sleep(2)

    print(f"Updated {update_cnt} posts, skipped {skip_cnt}")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(posts, f, indent=2)

        
# needs to be approved by Reddit
def praw_scraper():
    
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

def bright_data_proxy():
    # Bright Data proxy credentials
    proxy_host = "brd.superproxy.io"
    proxy_port = "33335"
    proxy_user = "brd-customer-hl_6be291c3-zone-datacenter_proxy1-country-us"
    proxy_pass = "vokn8vl52s87"

    proxy_url = f"http://{proxy_user}:{proxy_pass}@{proxy_host}:{proxy_port}"

    proxies = {
        "http": proxy_url,
        "https": proxy_url
    }

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    url = "https://www.reddit.com/r/shittysuperpowers/top.json?limit=10&t=all"

    response = requests.get(url, headers=headers, proxies=proxies, timeout=30)
    print(response.text)

    data = response.json()

    for post in data["data"]["children"]:
        title = post["data"]["title"]
        permalink = post["data"]["permalink"]
        print(title)
        print("https://www.reddit.com" + permalink)
        print()

def bright_data_api():
    url = "https://api.brightdata.com/request"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_TOKEN}"
    }

    payload = {
        "zone": ZONE,
        "url": SUBREDDIT_URL,
        "format": "raw"
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)

    print("Status:", response.status_code)
    print("Headers:", response.headers)

    if response.status_code != 200:
        print("Request failed:", response.status_code)
        return

    data = response.json()
    posts = data["data"]["children"]

    new_posts = []

    # check if post was seen
    for post in posts:
        post_data = post["data"]
        post_id = post_data["id"]

        if post_id not in seen_ids:
            new_posts.append(post_data)
            seen_ids.add(post_id)

    # write out
    if new_posts:
        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, "r") as f:
                existing = json.load(f)
        else:
            existing = []

        existing.extend(new_posts)

        with open(OUTPUT_FILE, "w") as f:
            json.dump(existing, f, indent=2)

        # update seen posts
        with open(STATE_FILE, "w") as f:
            json.dump(list(seen_ids), f)

    print(f"Collected {len(new_posts)} new posts.")

def main():
    start = time.time()
    scrape_subreddit()
    scrape_posts()
    print(f"Execution time: {time.time() - start:2f} seconds")
if __name__ == "__main__":
    main()