import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
from gensim.parsing.preprocessing import strip_multiple_whitespaces, strip_non_alphanum, strip_tags
import re

import string
import pke
from nltk.corpus import stopwords

import sys
sys.path.append("lexrank/")
from lexrank.lexrank import  degree_centrality_scores


# ==  Functions
def clean(sx):
    sx = strip_tags(sx)
    # ~ sx = strip_non_alphanum(sx)
    sx = re.sub(r'\n',' ',sx)
    sx = strip_multiple_whitespaces(sx)
    return sx



# ===========================================



# Load data  and embedder ============================

# ~ enty = pd.read_table('../data/concepts.tsv', names=['link','name','type','description'],header=0)
# ~ enty['clean_text']= enty["description"].apply(clean)
# ~ enty.drop_duplicates(subset=["name"], keep='first', inplace=True)

sbert = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
# ===========================================



top_k = 15

# ~ entid = 924
# ~ doc = enty.iloc[entid]['clean_text']
# ~ nm = enty.iloc[entid]['name']

nm = 'Self-driving_car'
f_in  = '../data/self-car.txt'
docs = []
with open(f_in, 'r') as fin:
    for dcc in fin:
        docs.append(dcc.strip('\r\n'))

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

if False:
    # Key-phrases by PKE (MultipartiteRank) =============

    # ~ print('Weighting candidates key-phrases')
    ext.candidate_weighting(threshold=0.74, method='average')

    # 5. get the 10-highest scored candidates as keyphrases
    keyphrases = ext.get_n_best(n=top_k)

    print("Key-phrases by PKE\n"+'-'*20)
    [print(kw) for kw in keyphrases ]

    print('='*20)
    #  ==================================================





# Key-phrases by SentBERT =============

# Extract Embeddings
doc_embedding = sbert.encode([doc])
word_embeddings = sbert.encode(words)

# Calculate distances and extract keywords
dists = util.pytorch_cos_sim(doc_embedding, word_embeddings).numpy()
idx_topK = dists.argsort()[0][::-1][:top_k]
kws = [(words[idx],dists[0][idx]) for idx in idx_topK]

# == Cut =================
thr=0.2
idx_cut = dists[0][:]>thr

wrc = [words[idx] for idx in range(len(words)) if idx_cut[idx]==True ]
wrc_e = word_embeddings[idx_cut]
dst_c = dists[0][idx_cut]

print('\n\n Key-phrases by SentBERT \n'+'-'*20)
[print(kw) for kw in kws ]

print('='*20)
#  ==================================================

if False:
    # Key-phrases by SentBERT+LexRank =============

    #Compute the pair-wise cosine similarities
    cos_scores = util.pytorch_cos_sim(word_embeddings, word_embeddings).numpy()

    #Compute the centrality for each sentence
    centrality_scores = degree_centrality_scores(cos_scores, threshold=None)

    #We argsort so that the first element is the sentence with the highest score
    most_central_sentence_indices = np.argsort(-centrality_scores)


    #Print the 5 sentences with the highest scores
    print("\n\n Key-phrases by SentBERT+LexRank\n"+'-'*20)
    for idx in most_central_sentence_indices[0:top_k]:
        print(words[idx].strip(), centrality_scores[idx])

    print('='*20)

    #  ==================================================


import matplotlib.pyplot as plt


import umap

tv = wrc_e

import hdbscan
ump = umap.UMAP(n_neighbors=15,n_components=5,metric='cosine') #,random_state=234
um = ump.fit_transform(tv)

# ~ um = tv

# ~ cos_scores = util.pytorch_cos_sim(word_embeddings, word_embeddings).numpy()

# ~ cls = hdbscan.HDBSCAN(min_cluster_size=10,metric='precomputed')
# ~ cls.fit(cos_scores)



cls = hdbscan.HDBSCAN(min_cluster_size=4,
                          metric='euclidean',
                          cluster_selection_method='eom').fit(um)

lbs  = cls.labels_
lbus = set(lbs)

plt.figure('UMAP')
plt.scatter(um[:, 0], um[:, 1],c = lbs) # ,c = lv_r, s=ts_r cmap=cmap


plt.show()





# ~ wc = [[(words[idx],dists[0][idx]) for idx in range(len(words)) if cls.labels_[idx]==lb] for lb in lbus  ]
wc = [[(wrc[idx],dst_c[idx]) for idx in range(len(wrc)) if cls.labels_[idx]==lb] for lb in lbus  ]
