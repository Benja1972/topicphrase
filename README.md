# Key-phrase extraction with Sentence Transformers
Simple code for extraction of key-phrases from single document based on Sentence Trasfomers [sentence-transformers](https://github.com/UKPLab/sentence-transformers). It combines several ideas from different pacakges. Core steps of pipe-line include:
- extract noun phrase candidates using spacy (for simplicity we take part of the  code from pke package [pke](https://github.com/boudinfl/pke));
- calculate embedding of phrases and orogonal document with help of  Sentence Trasfomers;
- cluster key-phrase vectors with HDBSCAN to group them in topics;
- sort groups (topics) and key-phases inside clusters by relevance to orogonal document.


```python
from src.key_phrase_Topic import *
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords
import nltk
import string

```

## Load data   


```python
f_in  = 'data/self-car.txt'
docs = []

with open(f_in, 'r') as fin:
    for dcc in fin:
        docs.append(dcc.strip('\r\n'))

doc = ' '.join(docs)
```

## Load pre-trained embedding model 


```python
model = 'distilbert-base-nli-stsb-mean-tokens'
sbert = SentenceTransformer(model)
```


```python
stop_words = stopwords.words('english')

pos = {'NOUN', 'PROPN', 'ADJ'} 
stoplist = list(string.punctuation)
stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
stoplist += stop_words
```

## Get noun phrases candidates


```python
words = get_candidates(doc,pos=pos,stoplist=stoplist)
```

    Selecting candidates key-phrases


## Calculate embedding of key-phrases and original text


```python
word_emb, dists = embed_sort(doc,words,sbert)
```

## Group candiadates in topics (clusters)


```python
lbs = get_clusters(word_emb)
```

## Sort groups by relevance to original document


```python
ws, rs = get_mean_sort(words,dists,lbs)
```

### Print key-phrases grouped in topics


```python
# Print =======================
print_top(ws[:15],n=5)
```

    [[('autonomous vehicles market', 0.5798149),
      ('autonomous vehicle', 0.57847726),
      ('autonomous car', 0.568246),
      ('autonomous vehicle industry', 0.5617635),
      ('many autonomous vehicles', 0.55915654)],
     [('self-driving car project', 0.52816546),
      ('Modern self-driving cars', 0.5205737),
      ('self-driving-car testing', 0.5061681),
      ('self-driving car industry', 0.50272787),
      ('self-driving car story', 0.49687785)],
     [('car navigation system', 0.48053306),
      ('vehicle control method', 0.47486627),
      ('auto-piloted car', 0.46408722),
      ('vehicle communication systems', 0.46348438),
      ('vehicle transmitting data', 0.4622034)],
     [('driver assistance technologies', 0.46776882),
      ('driver- assistance systems', 0.46307224),
      ('advanced driver-assistance systems', 0.4613356),
      ('advanced driver assistance systems', 0.44836435),
      ('Driving Systems', 0.4163338)],
     [('vehicle automation', 0.4749097),
      ('semi-automated vehicles', 0.44886082),
      ('first semi-automated car', 0.42419726),
      ('Automated cars', 0.4174018),
      ('automated vehicles', 0.39398754)],
     [('ultrasonic sensors', 0.46323648),
      ('sensory data', 0.4561283),
      ('sensory information', 0.43459147),
      ('electronic blind-spot assistance', 0.3792184),
      ('adaptive cruise control', 0.372184)],
     [('self-protective cars', 0.5074644),
      ('driver safety', 0.44420025),
      ('safety driver', 0.38207477),
      ('passenger privacy', 0.37961134),
      ('Traffic Safety', 0.37002492)],
     [('motion sensors', 0.44774908),
      ('surround cameras', 0.41574597),
      ('sensors', 0.3751243),
      ('Typical sensors', 0.34719855),
      ('sensor systems', 0.33258197)],
     [('human driver control', 0.4044003),
      ('human driver input', 0.36990845),
      ('robo-car', 0.3650677),
      ('Tesla Autopilot capability', 0.35918993),
      ('Tesla cars', 0.3484006)],
     [('driverless systems', 0.41181698),
      ('driverless technology', 0.3821989),
      ('Driverless vehicles', 0.3539668),
      ('Uber vehicle', 0.3418124),
      ('Uber test vehicle', 0.33859164)],
     [('robotic car', 0.43310237),
      ('Starsky Robotics', 0.37113926),
      ('robotic vehicles', 0.36968625),
      ('Robotics', 0.34612796),
      ('robot workers', 0.28748578)],
     [('individual cars', 0.424756),
      ('vehicle owners', 0.3330462),
      ('individual vehicle', 0.33175975),
      ('single drivers', 0.32401156),
      ('total private vehicle use', 0.31353214)],
     [('roadside real-time locating system', 0.42653123),
      ('location-based ads', 0.37391913),
      ('location system', 0.30351847),
      ('remote places', 0.2805937),
      ('surroundings', 0.2639191)],
     [('hybrid navigation', 0.33650905),
      ('navigation system', 0.33546335),
      ('inertial measurement units', 0.335176),
      ('GPS', 0.32566887),
      ('odometry', 0.29296926)],
     [('Autopilot technology', 0.36545154),
      ('Autopilot mode', 0.3173113),
      ('AV technology', 0.3170814),
      ('autopilot system', 0.2957281),
      ('DrivePilot', 0.2878706)]]



```python

```
