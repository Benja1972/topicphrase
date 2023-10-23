# Key-phrase extraction with Sentence Transformers
Simple code for extraction of key-phrases and group them in topics from a single document or set of documents based on dense vectors representations (embeddings). The Sentence Transformers [sentence-transformers](https://github.com/UKPLab/sentence-transformers) is used to embed the documents and key-phrases candidates. It combines several ideas from different packages. Core steps of pipe-line include:
- extract noun phrase candidates using spacy (for simplicity we take part of the  code from pke package [pke](https://github.com/boudinfl/pke));
- calculate embedding of phrases and original document with help of  Sentence Transformers;
- cluster key-phrase vectors with HDBSCAN to group them in topics (idea comes from nice [Top2Vec](https://github.com/ddangelov/Top2Vec));
- sort groups (topics) and key-phases inside clusters by relevance to original document.


```python
from src.key_topic import *
```

## Load data
For data we use Wikipedia article about [self-driving car](https://en.wikipedia.org/wiki/Self-driving_car)

```
    A self-driving car, also known as an autonomous vehicle (AV), connected and autonomous vehicle (CAV), full self-driving car or driverless car, or robo-car or robotic car, (automated vehicles and fully automated vehicles in the European Union) is a vehicle that is capable of sensing its environment and moving safely with little or no human input.
```

```python
f_in  = 'data/self-car.txt'
docs = []

with open(f_in, 'r') as fin:
    for dcc in fin:
        docs.append(dcc.strip('\r\n'))

doc = ' '.join(docs)
```

## Initiate key-phrases extractor


```python
ph = KeyPhraser()
```


## Fit documents


```python
kph.fit(doc)
```


## Sort groups by relevance to original document or centroids of clusters


```python
# Sort by similarity to centroind of cluster
wsr_c = kph.output_topn_topics()
    
# = Print topics  ==========================
print("\n\nSorted by centroids\n")
pprint(wsr_c)

```

    
```sh
Sorted by centroids

[(0.6345134,
  [('autonomous vehicle', 0.9286376),
   ('autonomous system', 0.92183506),
   ('autonomous driving', 0.92158717),
   ('autonomous car', 0.91708),
   ('autonomous transportation', 0.91210616)]),
 (0.5180938,
  [('self-driving car', 0.94790494),
   ('self-driving vehicle', 0.90880316),
   ('full self-driving car', 0.904925),
   ('self-driving mode %', 0.89748883),
   ('self-driving car industry', 0.89390194)]),
 (0.48395425,
  [('radar perception', 0.8261095),
   ('visual perception', 0.81868887),
   ('perception', 0.7913567),
   ('visual object recognition', 0.7782693),
   ('sensory information', 0.7743635)]),
 (0.45413223,
  [('automated driving', 0.95637953),
   ('automated driving equipment', 0.9389162),
   ('automated vehicles', 0.93130183),
   ('Automated cars', 0.92391855),
   ('automated-driving miles', 0.90534574)]),
 (0.45209697,
  [('driving tasks', 0.8971716),
   ('driving function', 0.89641565),
   ('driving features', 0.8827797),
   ('driver', 0.8584493),
   ('driving systems', 0.8579692)])]

```


```python
# Sort by similarity to original document
wsr_d = kph.doc_topn_topics(doc_id=0)

print("\n\nSorted by original doc\n")
pprint(wsr_d)
```

    
```sh
Sorted by original doc

[(0.6345134,
  ['self-driving car project',
   'Modern self-driving cars',
   'self-driving-car testing',
   'self-driving car industry',
   'self-driving car story']),
 (0.5180938,
  ['autonomous vehicles market',
   'autonomous vehicle',
   'autonomous car',
   'autonomous vehicle industry',
   'many autonomous vehicles']),
 (0.48395425,
  ['artificial intelligence',
   'automotive hacking',
   'smart systems',
   'Artificial Intelligence Laboratory',
   'system operator']),
 (0.45413223,
  ['private property',
   'personal ownership',
   'personal preferences',
   'licensed operator',
   'subsidiaries']),
 (0.45209697,
  ['self-interested preferences',
   'rational actors',
   'accurate detection',
   'high autonomy levels',
   'attentiveness'])]

```


### Granularity of clusters
Granularity of clusters could be controlled by decreasing `min_cluster_size`, the default is 10


```python
# Decrease 'min_cluster_size' to get more fine topics
kph = KeyPhraser(min_cluster_size = 6)

kph.fit(doc)
```

