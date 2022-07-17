import torch
from sklearn.manifold import TSNE
from numpy import reshape
import pandas as pd
import dataLoader
from openTSNE import TSNE

import matplotlib.pyplot as plt
dataset = dataLoader.LinkPredData("data/", "big_graph", use_subset=False)
data = dataset.load(transform=True)

"""
tsne = TSNE(
    perplexity=30,
    metric="euclidean",
    n_jobs=8,
    random_state=42,
    verbose=True,
)
data_x = data.x[0:data.x.shape[0]//2]
print(data_x.shape,data.x.shape)
embedding_train = tsne.fit(data_x)
print(embedding_train,embedding_train.shape)


tsne = TSNE(n_components=2, verbose=1, random_state=123)
z = tsne.fit_transform(data.x)
df = pd.DataFrame()
df["y"] = y
df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]

fig, ax = plt.subplots()
plt.tight_layout
plt.scatter(embedding_train[:,0],embedding_train[:,1],color="mediumslateblue")
plt.axis("off")
plt.savefig("plots/tsne.jpg")

plt.show()
"""
