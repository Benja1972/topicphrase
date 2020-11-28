import numpy as np
from sentence_transformers import SentenceTransformer, util

import string
import pke
from nltk.corpus import stopwords
import nltk
import spacy
from pprint import pprint


import umap
import hdbscan

# ==  Functions


def get_candidates(doc,pos,stoplist):
    nlp = spacy.load('en')  # or any model
    nlp.add_pipe(merge_compounds)
    ext = pke.unsupervised.MultipartiteRank()
    ext.load_document(doc, max_length=5173682, spacy_model=nlp)
    print('Selecting candidates key-phrases')
    ext.candidate_selection(pos=pos, stoplist=stoplist)
    words = [' '.join(cdd.surface_forms[0]) for st,cdd in ext.candidates.items()]
    return words


def get_clusters(word_emb, min_cluster_size=10):
    ump = umap.UMAP(n_neighbors=min_cluster_size,n_components=3,metric='cosine') #,random_state=234
    um = ump.fit_transform(word_emb)

    cls = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                            cluster_selection_epsilon=0.2,
                            min_samples=1,
                            metric='euclidean',
                            cluster_selection_method='eom').fit(um) # 'leaf' 'eom'

    lbs  = cls.labels_
    return lbs


def get_mean_sort(words,dists,lbs,n=5):
    lbus = set(lbs)
    wa = [[(words[idx],dists[0][idx]) for idx in range(len(words)) if lbs[idx]==lb] for lb in lbus]
    ws = [[wc[ix] for ix in np.argsort([w[1] for w in wc])[::-1]] for wc in wa]
    rs = [np.mean([wd[1] for wd in wi ][:n]) for wi in ws]
    idrs = np.argsort(np.array(rs))[::-1]
    ws = [ws[ids] for ids in idrs]
    return ws, rs
        

def embedd_dist(doc,words,sbert):
    # Extract Embeddings
    doc_emb = sbert.encode([doc])
    word_emb = sbert.encode(words)

    # Calculate distances 
    dists = util.pytorch_cos_sim(doc_emb, word_emb).numpy()

    return word_emb, dists


def merge_compounds(d):
    """ Merge compounds to be one token

    A compound is two tokens separated by a hyphen when the tokens are right next to the hyphen

    d (spacy.Doc): Document
    Returns: spacy.Doc

    > [t.text for t in nlp('peer-to-peer-to-peer')]
    ['peer', '-', 'to', '-', 'peer', 'to', '-', 'peer']
    > [t.text for t in merge_compounds(nlp('peer-to-peer-to-peer'))]
    ['peer-to-peer-to-peer']
    """
    # Returns beginning and end offset of spacy.Token
    offsets = lambda t: (t.idx, t.idx+len(t))

    # Identify the hyphens
    # for each token is it a hyphen and the next and preceding token are right next to the hyphen
    spans = [(i-1, i+1) for i in range(len(d))
             if i != 0 and i != len(d) and d[i].text == '-' and \
                offsets(d[i-1])[1] == offsets(d[i])[0] and \
                offsets(d[i+1])[0] == offsets(d[i])[1]
            ]
    # merging spans to account for multi-compound terms
    merged = []
    for i, (b, e) in enumerate(spans):
        # if the last spans ends when the current span begins,
        # merge those
        if merged and b == merged[-1][1]:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((b, e))

    # Merge the compounds in the document
    with d.retokenize() as retok:
        for b, e in merged:
            retok.merge(d[b:e+1], attrs={
                'POS': d[b].pos_,
                'LEMMA': ''.join(t.lemma_ for t in d[b:e+1])
            })
    return d

def print_top(wa,n=8):
    pprint([w[:n] for w in wa])
# ===========================================

if __name__ == '__main__':

    # Load data  and embedder ============================

    nm = 'Self-driving_car'
    print(nm)
    print('='*20)

    f_in  = '../data/self-car.txt'
    docs = []

    with open(f_in, 'r') as fin:
        for dcc in fin:
            docs.append(dcc.strip('\r\n'))

    doc = ' '.join(docs)

    model = 'distilbert-base-nli-stsb-mean-tokens'
    sbert = SentenceTransformer(model)
    # ===========================================


    # Get candidates =============
    stop_words = stopwords.words('english')

    pos = {'NOUN', 'PROPN', 'ADJ'} 
    stoplist = list(string.punctuation)
    stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
    stoplist += stop_words
    words = get_candidates(doc,pos=pos,stoplist=stoplist)

    #  Embded ================
    word_emb, dists = embedd_dist(doc,words,sbert)

    # Cluster ======================
    lbs = get_clusters(word_emb)

    # Group and sort===================
    ws, rs = get_mean_sort(words,dists,lbs,n=5)

    # Print =======================
    print_top(ws)
