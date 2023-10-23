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

[(37,
  0.6368047,
  [('autonomous vehicle', 0.91608304),
   ('autonomous driving', 0.9148414),
   ('autonomous car', 0.9091445),
   ('autonomous system', 0.9038766),
   ('autonomous transportation', 0.8886418)]),
 (7,
  0.5180938,
  [('self-driving car', 0.94790494),
   ('self-driving vehicle', 0.90880316),
   ('full self-driving car', 0.904925),
   ('self-driving mode %', 0.89748883),
   ('self-driving car industry', 0.89390194)]),
 (43,
  0.44183356,
  [('driver', 0.89716387),
   ('driver position', 0.8791572),
   ('driver experience', 0.86386347),
   ('driver seat', 0.8476667),
   ('Driver interactions', 0.8414953)]),
 (60,
  0.43026584,
  [('automotive vehicles', 0.9044908),
   ('automobiles', 0.89079916),
   ('cars', 0.8765358),
   ('vehicles', 0.8505863),
   ('motor vehicles', 0.8391547)]),
 (12,
  0.42813206,
  [('automated driving', 0.9292104),
   ('automated driving equipment', 0.9208274),
   ('automated vehicles', 0.89990115),
   ('automated-driving miles', 0.88774276),
   ('Automated cars', 0.8818551)])]


```


```python
# Sort by similarity to original document
wsr_d = kph.doc_topn_topics(doc_id=0)

print("\n\nSorted by original doc\n")
pprint(wsr_d)
```

    
```sh
Sorted by original doc

[(37,
  0.6368047,
  ['autonomous vehicles market',
   'autonomous vehicle',
   'autonomous car',
   'autonomous vehicle industry',
   'many autonomous vehicles']),
 (7,
  0.5180938,
  ['self-driving car project',
   'Modern self-driving cars',
   'self-driving-car testing',
   'self-driving car industry',
   'self-driving car story']),
 (43,
  0.44183356,
  ['driver assistance technologies',
   'advanced driver-assistance systems',
   'human driver control',
   'remote driver',
   'human driver input']),
 (60,
  0.43026584,
  ['car navigation system',
   'car sensors.citation',
   'auto-piloted car',
   'automotive vehicles',
   'vehicle system']),
 (12,
  0.42813206,
  ['automated driving equipment',
   'semi-automated vehicles',
   'automated-driving miles',
   'first semi-automated car',
   'Automated cars'])]


```


### Granularity of clusters
Granularity of clusters could be controlled by decreasing `min_cluster_size`, the default is 10


```python
# Decrease 'min_cluster_size' to get more fine topics
kph = KeyPhraser(min_cluster_size = 6)

kph.fit(doc)
```

