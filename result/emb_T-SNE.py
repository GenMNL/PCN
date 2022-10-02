from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# load data of csv
csv_path = os.path.join("./result", "chair", "emb.csv")
df = pd.read_csv(csv_path, index_col=0, )
data = df.values # transform value of dataframe to ndarray
labels = np.arange(1, 151)

# apply T-SNE to the emb dim
tsne = TSNE(n_components=2, perplexity=3)
data_reduced = tsne.fit_transform(data)

df = pd.DataFrame(data_reduced).T
df.to_csv("./result/chair/2d_emb.csv")

# visualizatin 2D plot
fig = plt.figure()
ax = fig.add_subplot(111)
sc = ax.scatter(data_reduced[:,0], data_reduced[:,1], c=labels, cmap='jet')
plt.axis('off')
plt.colorbar(sc)
plt.show()
