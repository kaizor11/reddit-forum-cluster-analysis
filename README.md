# reddit-forum-cluster-analysis
A python script that automatically scrapes the subreddit: r/shittysuperpowers for new posts and store them in a MongoDB Atlas database. We then cluster the data using []

auto.py is the main program and performs the following:
- scrape the r/shittysuperpowers subreddit for new posts and update old ones if they were skipped in previous iterations (due to Reddit scrape block)
- stores the scraped data in a MongoDB Atlas database
- cluters the data []

**Installing Dependencies** <br>
`pip install -r requirements.txt`

**Execution** <br>
`python auto.py x` where `x` is the time (in minutes) between each run (scrape -> store -> cluster)