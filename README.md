# reddit-forum-cluster-analysis
A python script that automatically scrapes the subreddit: r/shittysuperpowers for new posts and store them in a MongoDB Atlas database. We then cluster the data with KMeans and WordCloud

`auto.py` is the main program and performs the following:
- scrape the r/shittysuperpowers subreddit for new posts and update old ones if they were skipped in previous iterations (due to Reddit scrape block)
- stores the scraped data in a MongoDB Atlas database
- cluters the text data

**Setup** <br>
- Install dependencies using `pip install -r requirements.txt`
- Setup up a MongoDB Atlas database cluster at https://www.mongodb.com/products/platform/atlas-database. Put your MongoDB API key in a .env file

**Execution** <br>
`python auto.py x` where `x` is the time (in minutes) between each run (scrape -> store -> cluster)