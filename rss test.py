import feedparser
import requests
import pandas as pd
import time
import random

#CONFIGURATION
#Target Subreddit RSS URL
RSS_URL = "https://www.reddit.com/r/MachineLearning/new/.rss"

# USER-AGENT: Crucial for not getting blocked.
# Format: <Platform>:<AppID>:<Version> (by /u/<YourUsername>)
HEADERS = {
    'User-Agent': 'Script:MyResearchBot:v1.0 (by /u/Freeman97376)'
}

def fetch_rss_data(url):
    """
    Step 1: Fetch the list of latest posts using RSS.
    Returns a list of dictionaries with basic info.
    """
    print(f"[*] Fetching RSS feed from: {url}")
    feed = feedparser.parse(url, request_headers=HEADERS)

    if feed.status != 200:
        print(f"[!] Failed to fetch RSS. Status code: {feed.status}")
        return []

    posts = []
    for entry in feed.entries:
        posts.append({
            "title": entry.title,
            "link": entry.link,
            "published": entry.published,
            "content": ""  # Placeholder for step 2
        })
    
    print(f"[*] Successfully retrieved {len(posts)} posts from RSS.")
    return posts

def fetch_post_content(url):
    """
    Step 2: Fetch the full body text (selftext) of a post using the JSON API.
    """
    # Append .json to the URL to get raw data
    json_url = url.rstrip('/') + ".json"
    
    try:
        response = requests.get(json_url, headers=HEADERS, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            # Navigate the JSON structure to find the post body
            # structure: [listing_data, comment_data] -> listing -> children -> post -> data
            post_data = data[0]['data']['children'][0]['data']
            return post_data.get('selftext', '')
        elif response.status_code == 429:
            print("[!] Rate limited (429). Sleeping for 10s...")
            time.sleep(10)
            return ""
        else:
            print(f"[!] Error {response.status_code} for URL: {url}")
            return ""
            
    except Exception as e:
        print(f"[!] Exception fetching content: {e}")
        return ""

def main():
    #  Get the list of posts using rss
    posts = fetch_rss_data(RSS_URL)
    
    if not posts:
        print("[!] No posts found. Exiting.")
        return

    print(f"[*] Starting detailed scrape for {len(posts)} posts...")
    
    # Iterate through posts  links to get content
    for i, post in enumerate(posts):
        print(f"    Processing {i+1}/{len(posts)}: {post['title'][:40]}...")
        
        # Fetch content
        content = fetch_post_content(post['link'])
        
        #Handle cases where content is empty 
        if not content:
            post['content'] = "[Link Only / No Text]"
        else:
            post['content'] = content
            
        # RATE LIMITING: Random sleep to mimic human behavior
        time.sleep(2 + random.random() * 2)

    #  Save to CSV using Pandas
    df = pd.DataFrame(posts)
    output_file = "reddit_full_data.csv"
    
    
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    
    print("-" * 30)
    print(f"[*] Done! Data saved to: {output_file}")
    print(df.head())

if __name__ == "__main__":
    main()