import sys
import time
from datasets import load_dataset
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import tokenize
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import json
import string
from nltk.corpus import stopwords
import nltk
import mplcursors as mpl
import numpy as np
from MongoDBconnection import MongoDBConnection
from scraper import scrape_subreddit, scrape_posts, SUBREDDIT_URL_OLD
from wordcloud import WordCloud, STOPWORDS
import math
nltk.download('stopwords')


scrape_interval_min = int(sys.argv[1])

def scrape():
    print(f"Scraping from {SUBREDDIT_URL_OLD}...")

    start = time.time()
    scrape_subreddit()
    print(f"Execution time: {time.time() - start:2f} seconds")

    start = time.time()
    scrape_posts()
    print(f"Execution time: {time.time() - start:2f} seconds")

def store():
    print("store data in cloud ")
    mongo = MongoDBConnection()

    mongo.upload_json_file("posts.json")

    print(f"currently has: {mongo.collection.count_documents({})} instances of data")

def retrieve_data():
    mongo = MongoDBConnection()
    return mongo.get_data()

def preprocess(df):
    print(df.head())
    data = df.iloc[:,0]
    dataset = []
    for entry in data:
        temp = entry.get("title", "") + " " + entry.get("body", "")

        #remove the r/shittysuperpowers end tag
        temp = temp.replace("\n", " ")
        temp = temp.replace("Tell the internet how you will prevent bank robberies and save lives with the worst superpowers in the world here at  r/shittysuperpowers . Be creative and don't hold back. No superpower too awful.   Happy Shitting!", "")
        
        temp = temp.strip() # remove extra whitespace
        temp = temp.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
        dataset.append(temp)
    return dataset

def preprocess_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f) 

    dataset = []
    for entry in data:
        temp = entry.get("title", "") + " " + entry.get("body", "")

        #remove the r/shittysuperpowers end tag
        temp = temp.replace("\n", " ")
        temp = temp.replace("Tell the internet how you will prevent bank robberies and save lives with the worst superpowers in the world here at  r/shittysuperpowers . Be creative and don't hold back. No superpower too awful.   Happy Shitting!", "")
        
        temp = temp.lower() # make str lowercase
        temp = temp.strip() # remove extra whitespace
        temp = temp.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
        dataset.append(temp)
    return dataset

def cluster(in_docs):
    tokens = [list(tokenize(x, to_lower=True)) for x in in_docs]

    #grab stopwords corpus
    stops = set(stopwords.words('english'))

    #there are some extremely common words in r/shittysuperpowers that don't give much context
    addtl_words = ["ability",
                   "someone",
                   "make",
                   "time", # true that there could be time-based superpowers but often accompanied by "rewind", "stop", etc. 
                   "every",
                   "power",
                   "people",
                   "anyone"]
    for word in addtl_words:
        stops.add(word)

    docs = [[token for token in doc if token not in stops] for doc in tokens]

    inset = [TaggedDocument(doc, [i]) for i, doc in enumerate(docs)]

    # generate embeddings of our texts
    model = Doc2Vec(inset, 
                    vector_size=5, 
                    window=2, 
                    min_count=1,
                    workers=4)

    # turn texts into vectors by using the model
    out = [model.infer_vector(x) for x in docs]
    #print(out[:5])

    # define and fit cluster model
    cluster_model = KMeans(n_clusters=5,
                        random_state=560
                            ).fit(out)
    print("Cluster Centers:")
    print(cluster_model.cluster_centers_)

    # predict clusters
    preds = cluster_model.predict(out)
    points_5d = [list(item) for item in out]
    points_x = [point[0] for point in points_5d]
    points_y = [point[1] for point in points_5d]

    # create a lookup dict based on the first two dimensions 
    # (probably deterministic enough, and mplcursor doesn't support 3+ dimensions)
    lookup_x = {}
    for ind, val in enumerate(points_x):
        val = round(np.float64(val), 4)
        if lookup_x.get(val, -1) == -1:
            lookup_x[val] = [ind]
        else:
            lookup_x[val].append(ind)

    lookup_y = {}
    for ind, val in enumerate(points_y):
        val = round(np.float64(val), 4)
        if lookup_y.get(val, -1) == -1:
            lookup_y[val] = [ind]
        else:
            lookup_y[val].append(ind)
    #print(lookup_x)
    
    # sample viz showing just first 3 dims
    def display_annotations(cursor):
        xinds = lookup_x[round(cursor.target[0], 4)]
        yinds = lookup_y[round(cursor.target[1], 4)]

        doc_ind = 0
        for ind in xinds:
            if ind in yinds:
                doc_ind = ind
        
        cursor.annotation.set_text(f'{in_docs[doc_ind]}')


    fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    plt.scatter(points_x, 
                points_y, 
                #points_z, 
                c=preds
                )
    plt.tight_layout()

    cursor = mpl.cursor(hover=True)
    cursor.connect("add", display_annotations)
    plt.show()

    # predict clusters
    return preds, [" ".join(doc) for doc in docs]

def plot_wordclouds(df, text_col, cluster_col):
    if df.empty: return
    
    clusters = sorted(df[cluster_col].unique())
    n_clusters = len(clusters)
    cols = 3
    rows = math.ceil(n_clusters / cols)
    
    plt.figure(figsize=(20, 5 * rows))

    for i, c_id in enumerate(clusters):
        # Join text for this cluster
        text = " ".join(df[df[cluster_col] == c_id][text_col].astype(str))
        
        # Generate cloud
        wc = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS).generate(text)
        
        # Plot
        plt.subplot(rows, cols, i + 1)
        plt.imshow(wc, interpolation="bilinear")
        plt.title(f"Cluster {c_id}", fontsize=14)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("cluster_result.png") 
    print("Wordcloud saved as 'cluster_result.png'")
    plt.show()



def main():

    # while True:
    for i in range(1):
        #scrape data
        scrape()

         #store data
        #store()

        # retrieve data
        #data_in = retrieve_data()

        # read data
        input = preprocess_json("posts.json")

        # cluster data
        cluster(input)
        preds, out_docs = cluster(input)
        

        print("Preparing visualization...")
        df = pd.DataFrame({
            'title': out_docs,   
            'cluster_id': preds    
        })
        print("Generating wordclouds")
        plot_wordclouds(df, text_col='title', cluster_col='cluster_id')
        
        print(f"Sleeping for {scrape_interval_min} minutes")
        time.sleep(scrape_interval_min * 60)

if __name__ == "__main__":
    main()