import numpy as np
from sentence_transformers import SentenceTransformer, util

import string
import pke

import spacy
from spacy.language import Language
from pprint import pprint


import umap
import hdbscan

# ==  Functions
def get_candidates(doc,pos = {'NOUN', 'PROPN', 'ADJ'},stoplist = None):
    nlp = spacy.load('en_core_web_sm')  # or any model
    nlp.add_pipe("merge_compound")
    ext = pke.unsupervised.MultipartiteRank()
    stop_list = list(string.punctuation)
    stop_list += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
    stop_list += pke.lang.stopwords.get('en')
    if stoplist:
        stop_list += stoplist
    # ~ ext.load_document(doc, max_length=5173682, spacy_model=nlp)
    ext.load_document(doc, spacy_model=nlp, stoplist=stoplist)
    print('Selecting candidates key-phrases')
    # ~ ext.candidate_selection(pos=pos, stoplist=stoplist)
    ext.candidate_selection(pos=pos)
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


@Language.component('merge_compound')
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

def centr_sort(words, word_emb, lbs, doc_emb, sort_by="centroid"):
    wn = []
    for lc in set(lbs):
        si = list(np.where(lbs==lc)[0])
        cle = word_emb[si]
        word_cls = [words[i] for i in si]

        ccl = cle.mean(axis=0)
        dsc = util.pytorch_cos_sim(ccl, doc_emb).numpy()[0][0]
        
        if sort_by=="doc":
            sc = util.pytorch_cos_sim(doc_emb, cle).numpy()
        else:
            sc = util.pytorch_cos_sim(ccl, cle).numpy()
        
        idx = np.argsort(sc)[0][::-1]
        
        wnn = (dsc, [word_cls[i] for i in idx])
        wn.append(wnn)
    
    smar = np.array([w[0] for w in wn])
    idxs = np.argsort(smar)[::-1]

    wns = [wn[ids] for ids in idxs]
    return wns
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
  
    # Add custom stoplist
    # ~ stoplist = []

    words = get_candidates(doc)

    #  Embded ================
    word_emb = sbert.encode(words)
    doc_emb = sbert.encode([doc])

    # Cluster ======================
    lbs = get_clusters(word_emb,6)

    # Group and sort 
    wsr_c = centr_sort(words, word_emb, lbs, doc_emb)
    wsr_d = centr_sort(words, word_emb, lbs, doc_emb, sort_by="doc")


    print("\n\nSorted by centroids\n")
    for w in wsr_c:
        print(w[0], w[1][:3])

    print("\n\nSorted by original doc\n")
    for w in wsr_d:
        print(w[0], w[1][:3])
