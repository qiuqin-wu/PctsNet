import pandas as pd
import networkx as nx

ppi_data = pd.read_csv("/fs/ess/PCON0022/wangqi/WQQ/CCAN/data/cell_line_scGNN/protein_protein_zuizhong.txt", sep="\t")

G = nx.from_pandas_edgelist(ppi_data, 'protein1', 'protein2')

from node2vec import Node2Vec
node2vec = Node2Vec(G, dimensions=128, walk_length=80, num_walks=10, workers=4)
model = node2vec.fit()
protein_embeddings = model.wv
all_embeddings = protein_embeddings.vectors
print("all_embeddings.shape:",all_embeddings.shape)

model.wv.save_word2vec_format("/fs/ess/PCON0022/wangqi/WQQ/CCAN/data/cell_line_scGNN/node2vec_embeddings.txt")