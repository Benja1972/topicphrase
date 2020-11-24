import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
from gensim.parsing.preprocessing import strip_multiple_whitespaces, strip_non_alphanum, strip_tags
import re

import string
import pke
from nltk.corpus import stopwords



# ==  Functions
def clean(sx):
    sx = strip_tags(sx)
    # ~ sx = strip_non_alphanum(sx)
    sx = re.sub(r'\n',' ',sx)
    sx = strip_multiple_whitespaces(sx)
    return sx



# ===========================================



# Load data  and embedder ============================

nm = 'Self-driving_car'
f_in  = '../data/self-car.txt'
docs = []
with open(f_in, 'r') as fin:
    for dcc in fin:
        docs.append(dcc.strip('\r\n'))

sbert = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
# ===========================================



top_k = 15
doc = ' '.join(docs)

# Get candidates =============
stop_words = stopwords.words('english')

pos = {'NOUN', 'PROPN', 'ADJ'} #"ADV"
stoplist = list(string.punctuation)
stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
stoplist += stop_words

ext = pke.unsupervised.MultipartiteRank()
ext.load_document(doc, max_length=5173682)
print('Selecting candidates key-phrases')
ext.candidate_selection(pos=pos, stoplist=stoplist)
words = [' '.join(cdd.surface_forms[0]) for st,cdd in ext.candidates.items()]
#  ===========================

print('='*20)
print(nm)
print('='*20)







# Key-phrases by SentBERT =============

# Extract Embeddings
doc_emb = sbert.encode([doc])
word_emb = sbert.encode(words)

# Calculate distances 
dists = util.pytorch_cos_sim(doc_emb, word_emb).numpy()


# Sorting
idxs = dists.argsort()[0][::-1]
kws = [(words[idx],dists[0][idx]) for idx in idxs]







import umap



import hdbscan
ump = umap.UMAP(n_neighbors=10,n_components=3,metric='cosine') #,random_state=234
um = ump.fit_transform(word_emb)



cls = hdbscan.HDBSCAN(min_cluster_size=15,
                        cluster_selection_epsilon=0.2,
                        min_samples=1,
                        metric='euclidean',
                        cluster_selection_method='leaf').fit(um) # 'leaf' 'eom'

lbs  = cls.labels_
lbus = set(lbs)



def get_mean_sort(wa,n=5):
    ws = [[wc[ix] for ix in np.argsort([w[1] for w in wc])[::-1]] for wc in wa]
    rs = [np.mean([wd[1] for wd in wi ][:n]) for wi in ws]
    idrs = np.argsort(np.array(rs))[::-1]
    ws = [ws[ids] for ids in idrs]
    return ws, rs
        



wc = [[(words[idx],dists[0][idx]) for idx in range(len(words)) if lbs[idx]==lb] for lb in lbus]
ws, rs = get_mean_sort(wc,n=5)


from pprint import pprint
def print_top(wa,n=8):
    pprint([w[:n] for w in wa])




# Plot
import matplotlib.pyplot as plt
import seaborn as sns



pltte = sns.color_palette('Paired', 100)
cls = [pltte[x] if x >= 0 else (0.5, 0.5, 0.5) for x in lbs]

plt.figure('UMAP')
plt.scatter(um[:, 0], um[:, 1],c = cls,s=1) # ,c = lv_r, s=ts_r cmap=cmap


plt.show()
