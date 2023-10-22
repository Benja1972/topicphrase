import numpy as np
from sentence_transformers import SentenceTransformer, util

import string
import pke
from collections import Counter

import spacy
from spacy.language import Language
from spacy.tokens import Doc
from pprint import pprint


import umap
import hdbscan



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


# Package class
class KeyPhraser:
    def __init__(self, 
                 model = 'distilbert-base-nli-stsb-mean-tokens', 
                 stoplist = None, 
                 grammar = "NP: {<ADJ>*<NOUN|PROPN>+}",
                 tokenizer = "grammar",
                 pos = {'NOUN', 'PROPN', 'ADJ'},
                 maximum_word_number=3):
        
        nlp = spacy.load('en_core_web_sm')
        nlp.add_pipe("merge_compound")
        # add the sentence splitter
        nlp.add_pipe('sentencizer')
        self.nlp = nlp
        
        self.embedder = SentenceTransformer(model)
        self.grammar =  grammar
        self.pos = pos
        self.maximum_word_number=maximum_word_number

        self.stoplist = list(string.punctuation)
        self.stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        self.stoplist += pke.lang.stopwords.get('en')
        if stoplist:
            self.stoplist += stoplist
        
        if tokenizer == "grammar":
            self.extractor = pke.unsupervised.PositionRank()
        elif tokenizer == "POS":
            self.extractor = pke.unsupervised.MultipartiteRank()
        else:
            raise ValueError(f'Invalid tokenizer: {tokenizer}. Select one of ["POS", "grammar"]')
        
        self.tokenizer = tokenizer
        
    def load_document(self, text):
        print('Loading documents\n'+40*'-')
        if (isinstance(text, list) and all(isinstance(doc, str) for doc in text)):
            inputs = Doc.from_docs([self.nlp(doc) for doc in text])
            docs = text
        elif  isinstance(text, str):
            inputs = text
            docs = [text]
        else:
            raise ValueError(f'Input documents should be a string or list of stings!')

        self.extractor.load_document(inputs, spacy_model=self.nlp, stoplist=self.stoplist)
        self.docs = docs
    
    def get_candidates(self):
        print('Selecting candidates key-phrases\n'+40*'-')
        if self.tokenizer == "grammar":
            self.extractor.candidate_selection(grammar=self.grammar, 
                                               maximum_word_number=self.maximum_word_number)
        elif self.tokenizer == "POS":
            self.extractor.candidate_selection(pos=self.pos)
        else:
            raise ValueError(f'Invalid tokenizer: {self.tokenizer}. Select one of ["POS", "grammar"]')
        
        self.vocab = [' '.join(cdd.surface_forms[0]) for st,cdd in self.extractor.candidates.items()]
    
    def __embedding(self):
        print('Embedding documents and phrases \n'+40*'-')
        # embedding 
        self.vocab_emb = self.embedder.encode(self.vocab)
        self.doc_emb = self.embedder.encode(self.docs)

    def __cluster(self):
        # ~ self.__embedding()
        print('Clustering \n'+40*'-')
        
        ump = umap.UMAP(n_neighbors=self.min_cluster_size,
                        n_components=3,
                        metric='cosine') #,random_state=234
        um = ump.fit_transform(self.vocab_emb)

        cls = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size,
                                cluster_selection_epsilon=self.cluster_selection_epsilon,
                                min_samples=1,
                                metric='euclidean',
                                cluster_selection_method='eom').fit(um) # 'leaf' 'eom'

        self.labels  = cls.labels_
    
    def topic_modeling(self, min_cluster_size=10, cluster_selection_epsilon=0.2):
        self.min_cluster_size = min_cluster_size
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.__embedding()
        self.__cluster()
        
        # == Calculate topics == 
        
        
        # Topic counts
        cnt = Counter(self.labels)
        counts = dict(cnt.most_common())
        
        # Topic calculus
        topic_sats = {lc:[(self.vocab[i],self.vocab_emb[i]) for i in list(np.where(self.labels==lc)[0])] for lc in self.labels}
        topic_sats = {lc:self.__sort_centroid(v) for lc,v in topic_sats.items()}
        
        # Topic embeddings and sorted words
        # ~ topics_embeddings = {lc:self.vocab_emb[list(np.where(self.labels==lc)[0])].mean(axis=0) for lc in set(self.labels)}
        
        topics_words = {lc:v[1] for lc,v in topic_sats.items()}
        topics_embeddings = {lc:v[0] for lc,v in topic_sats.items()}
        # ~ vs = np.vstack([v[0] for v in topic_words.items()])
        vs = np.vstack(list(topics_embeddings.values()))
        doc_topic_matrix = util.pytorch_cos_sim(self.doc_emb, vs).numpy()
        
        # Documents-topics similarity distribution 
        self.doc_topic_matrix = doc_topic_matrix
        
        sim = doc_topic_matrix.mean(axis=0)
        
        # Similarity between documents and centroids of topics 
        topic_doc_sim = {lc:sim[i] for i,lc in enumerate(list(topics_embeddings.keys()))}
        
        # Store topics in the class variable
        topics = {lc: {"topic_words":topics_words[lc],
                        "embedding":topics_embeddings[lc],
                        "counts": counts[lc],
                        "doc_similarity":topic_doc_sim[lc]} for lc in topics_embeddings.keys()}
        self.topics = dict(sorted(topics.items(), key=lambda item: item[1]["doc_similarity"],reverse= True))
    
    
    def print_topn_topics(self, top_n = 5, top_n_words = 5):
        out = [(self.topics[ls]["doc_similarity"],self.topics[ls]["topic_words"][:top_n_words]) for ls in list(self.topics.keys())[:top_n]]
        pprint(out)
        
    
    @staticmethod
    def __sort_centroid(wd):
        class_emb = np.vstack([w[1] for w in wd])
        centr_emb = class_emb.mean(axis=0)
        sc = util.pytorch_cos_sim(centr_emb, class_emb).numpy()
        idx = np.argsort(sc)[0][::-1]
                
        return centr_emb, [(wd[i][0],sc[0][i]) for i in idx]


    def doc_topn_topics(self, doc_id = None, top_n = 5, top_n_words = 5):
        topic_id = dict(enumerate(list(self.topics.keys())))
        doc_emb = self.doc_emb[doc_id]
        
        tp_v = self.doc_topic_matrix[doc_id]
        idx = np.argsort(tp_v)[::-1]
        idx = idx[:top_n]
        wn = []
        for i in idx:
            lc = topic_id[i]
            
            si = list(np.where(self.labels==lc)[0])
            cle = self.vocab_emb[si]
            word_cls = [self.vocab[i] for i in si]

            ccl = cle.mean(axis=0)
            sc = util.pytorch_cos_sim(doc_emb, cle).numpy()
            idx_w = np.argsort(sc)[0][::-1]
            idx_w = idx_w[:top_n_words]
            
            wnn = (tp_v[i], [word_cls[i] for i in idx_w])
            wn.append(wnn)
        return wn
            

        

    def sorted_topics(self, sort_by="centroid", doc_id = 0):
        wn = []
        for lc in set(self.labels):
            si = list(np.where(self.labels==lc)[0])
            cle = self.vocab_emb[si]
            word_cls = [self.vocab[i] for i in si]

            ccl = cle.mean(axis=0)
            doc_emb = self.doc_emb[doc_id]
            dsc = util.pytorch_cos_sim(ccl, doc_emb).numpy()[0][0]
            
            if sort_by=="doc":
                sc = util.pytorch_cos_sim(doc_emb, cle).numpy()
            elif sort_by=="centroid":
                sc = util.pytorch_cos_sim(ccl, cle).numpy()
            else:
                raise ValueError(f'Sorting methond: {sort_by} is not implemented. Select one of ["centroid", "doc"]')
            
            idx = np.argsort(sc)[0][::-1]
            
            wnn = (dsc, [word_cls[i] for i in idx])
            wn.append(wnn)
        
        smar = np.array([w[0] for w in wn])
        idxs = np.argsort(smar)[::-1]

        return [wn[ids] for ids in idxs]
        
    def fit(self,docs):
        self.load_document(docs)
        self.get_candidates()
        self.topic_modeling()
        


# ===========================================

if __name__ == '__main__':

    # = Load data  ===============================

    f_in  = '../data/self-car.txt'
    docs = []

    with open(f_in, 'r') as fin:
        for dcc in fin:
            docs.append(dcc.strip('\r\n'))

    # ~ doc = ' '.join(docs[:5])
    doc = docs[:15]

    
    # = Initiate key-phrases extractor ===========
    kph = KeyPhraser()
    
    # = Fit document ============================
    kph.fit(doc)


    # = Get sorted topics  ======================
    wsr_d = kph.sorted_topics(sort_by="doc")
    wsr_c = kph.sorted_topics(sort_by="centroid")
    
    # = Print topics  ==========================
    print("\n\nSorted by centroids\n")
    for w in wsr_c:
        print(w[0], w[1][:3])

    print("\n\nSorted by original doc\n")
    for w in wsr_d:
        print(w[0], w[1][:3])
