from datasets import load_dataset
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from .utils import tokenize
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

# load dataset
ds = load_dataset("FutureMa/EvasionBench", split="train")
docs = list(ds["answer"])

# first tokenize our text
docs = [list(tokenize(x, to_lower=True)) for x in docs]
inset = [TaggedDocument(doc, [i]) for i, doc in enumerate(docs)]

#print(inset[3])

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
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
plt.scatter(points_x, points_y, points_z, c=preds)
plt.tight_layout()
plt.show()