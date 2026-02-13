import sys
import time
from datasets import load_dataset
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import tokenize
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import json

from scraper import scrape_subreddit, scrape_posts, SUBREDDIT_URL_OLD

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
    pass

def preprocess(json_file):
    # TODO: replace json_file input with db input (or add as option)
    with open(json_file, "r") as f:
        data = json.load(f)

    dataset = []
    for entry in data:
        temp = entry.get("title", "") + " " + entry.get("body", "")
        # TODO: more robust preprocessing of strings
        temp = temp.strip()
        dataset.append(temp)
    return dataset

def cluster(docs):
    docs = [list(tokenize(x, to_lower=True)) for x in docs]
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
    points_z = [point[2] for point in points_5d]

    # sample viz showing just first 3 dims
    # TODO: resolve negative sqrt error
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plt.scatter(points_x, points_y, points_z, c=preds)
    plt.tight_layout()
    plt.show()

    # predict clusters
    return preds


def main():

    while True:
        # scrape data
        scrape()

        # store data
        # store()

        # read data
        # input = preprocess("posts.json")

        # cluster data
        # cluster(input)

        print(f"Sleeping for {scrape_interval_min} minutes")
        time.sleep(scrape_interval_min * 60)

if __name__ == "__main__":
    main()