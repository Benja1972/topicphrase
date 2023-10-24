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
Extracted topics ranked by similarity to the whole corpus and phrased sorted by centroids

[(6,
  0.6340678,
  [('autonomous vehicle', 0.9270642),
   ('autonomous driving', 0.9261311),
   ('autonomous car', 0.92045784),
   ('autonomous system', 0.91655105),
   ('autonomous transportation', 0.9070121)]),
 (3,
  0.5180938,
  [('self-driving car', 0.94790494),
   ('self-driving vehicle', 0.90880316),
   ('full self-driving car', 0.904925),
   ('self-driving mode %', 0.89748883),
   ('self-driving car industry', 0.89390194)]),
 (27,
  0.4600281,
  [('radar perception', 0.8565458),
   ('visual perception', 0.83902097),
   ('perception', 0.8050251),
   ('visual object recognition', 0.7988533),
   ('computer vision', 0.78779423)]),
 (12,
  0.44258612,
  [('automotive vehicles', 0.91408443),
   ('automobiles', 0.8592148),
   ('car manufacturers', 0.8539634),
   ('cars', 0.85164714),
   ('automotive industry', 0.84958434)]),
 (11,
  0.43725145,
  [('driving function', 0.9042318),
   ('driving tasks', 0.89559156),
   ('driving features', 0.8852626),
   ('driving', 0.8605223),
   ('driving systems', 0.8573065)])]

```


```python
# Sort by similarity to original document
wsr_d = kph.doc_topn_topics(doc_id=0)

print("\n\nSorted by original doc\n")
pprint(wsr_d)
```

    
```sh
Extracted topics and phrases ranked by similarity to the doc 0 in the corpus

[(6,
  0.6340678,
  [('autonomous vehicles market', 0.583479),
   ('autonomous vehicle', 0.5822828),
   ('autonomous car', 0.57185763),
   ('autonomous vehicle industry', 0.56481636),
   ('many autonomous vehicles', 0.55988705)]),
 (3,
  0.5180938,
  [('self-driving car project', 0.52936566),
   ('modern self-driving cars', 0.5236278),
   ('self-driving-car testing', 0.50891966),
   ('self-driving car industry', 0.50497735),
   ('self-driving car story', 0.49697512)]),
 (27,
  0.4600281,
  [('ultrasonic sensors', 0.4619211),
   ('sensory data', 0.45013982),
   ('sensory information', 0.4288388),
   ('electronic blind-spot assistance', 0.38294083),
   ('visual object recognition', 0.36293906)]),
 (12,
  0.44258612,
  [('car navigation system', 0.4831993),
   ('vehicle control method', 0.47435156),
   ('car sensors.citation', 0.46583062),
   ('vehicle communication systems', 0.46028528),
   ('vehicle control', 0.45426378)]),
 (11,
  0.43725145,
  [('driver assistance technologies', 0.46755123),
   ('auto-piloted car', 0.46486485),
   ('advanced driver-assistance systems', 0.45932925),
   ('driving systems', 0.41803557),
   ('vehicle ai', 0.4142478)])]

```


### Granularity of clusters
Granularity of clusters could be controlled by decreasing `min_cluster_size`, the default is 10


```python
# Decrease 'min_cluster_size' to get more fine topics
kph = KeyPhraser(min_cluster_size = 6)

kph.fit(doc)
```

