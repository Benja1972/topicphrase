# Key-phrase extraction with Sentence Transformers
Simple code for extraction of key-phrases from single document based on Sentence Trasfomers [sentence-transformers](https://github.com/UKPLab/sentence-transformers). It combines several ideas from different pacakges. Core steps of pipe-line include:
- extract noun phrase candidates using spacy (for simplicity we take part of the  code from pke package [pke](https://github.com/boudinfl/pke));
- calculate embedding of phrases and orogonal document with help of  Sentence Trasfomers;
- cluster key-phrase vectors with HDBSCAN to group them in topics;
- sort groups (topics) and key-phases inside clusters by relevance to orogonal document.


```python
from src.key_topic import *
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords
import nltk
import string
```

## Load data   
For data we use Wikipedia article about [self-driving car](https://en.wikipedia.org/wiki/Self-driving_car)


    A self-driving car, also known as an autonomous vehicle (AV), connected and autonomous vehicle (CAV), full self-driving car or driverless car, or robo-car or robotic car, (automated vehicles and fully automated vehicles in the European Union) is a vehicle that is capable of sensing its environment and moving safely with little or no human input.


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
word_emb = sbert.encode(words)
doc_emb = sbert.encode([doc])
```

## Group candiadates in topics (clusters)


```python
lbs = get_clusters(word_emb)
```

## Sort groups by relevance to original document or centroinds of clusters


```python
# Sort by similarity to centroind of cluster
wsr_c = centr_sort(words, word_emb, lbs, doc_emb)

print("\n\nSorted by centroids\n")
for w in wsr_c:
    print(w[0], w[1][:5])
```

    
    
    Sorted by centroids
    
    0.62424296 ['autonomous driving', 'autonomous vehicle', 'autonomous car', 'autonomous system', 'autonomous transportation']
    0.5434958 ['self-driving car', 'self-driving car industry', 'self-driving vehicles', 'self-driving', 'self-driving car project']
    0.4824011 ['visual perception', 'radar perception', 'perception', 'visual object recognition', 'computer vision']
    0.4445539 ['vehicle system', 'vehicles', 'vehicle types', 'vehicle technology', 'motor vehicles']
    0.44330907 ['driver', 'driving task', 'driver position', 'Driving Systems', 'driver seat']
    0.4168215 ['automotive vehicles', 'automotive industry', 'car manufacturers', 'automotive', 'car makers']
    0.39603418 ['automation', 'automation technology', 'automated vehicles', 'automation level', 'automated features']
    0.38521442 ['sensors', 'sensor systems', 'sensor-based infrastructure', 'surveillance apparatus', 'surveillance']
    0.3617182 ['navigation system', 'GPS', 'GPS position', 'appropriate navigation paths', 'satellite navigation system']
    0.34224677 ['Robotics', 'Starsky Robotics', 'robotic vehicles', 'robotic car', 'robotics ordinance']
    0.34062952 ['Google driverless car', 'driverless technology', 'driverless car', 'driverless', 'driverless systems']
    0.33944237 ['safety', 'safety hazards', 'public safety', 'safety issues', 'protection']
    0.33884573 ['Autopilot mode', 'autopilot', 'autopilot system', 'Autopilot technology', 'DrivePilot']
    0.3067453 ['individual vehicle', 'personal ownership', 'owner', 'subsidiaries', 'individual cars']
    0.27357867 ['artificial intelligence', 'Artificial Intelligence Laboratory', 'hacking', 'software hacking', 'probabilistic machine learning vehicle AI']
    0.26621902 ['location', 'surroundings', 'places', 'person', 'recreational areas']
    0.26551753 ['communication systems', 'communication', 'communication networks', 'communication connection', 'information']
    0.25277805 ['devices', 'relevant', 'information available', 'manner', 'localization']
    0.25201482 ['fuel efficiency improvement', 'fuel economy', 'fuel savings', 'higher fuel efficiency', 'fuel']
    0.23806205 ['software systems', 'software', 'operating software', 'computer programs', 'operating system']
    0.23448318 ['ethical issues', 'ethical problems', 'moral basis', 'ethical theories', 'ethical preferences']
    0.20482199 ['Road Traffic', 'traffic', 'traffic-jam', 'traffic accidents', 'traffic collisions']
    0.20016664 ['interior design', 'interior design industry', 'buildings', 'housing', 'architecture']
    0.19620562 ['energy consumption', 'electricity consumption', 'electricity', 'energy', 'global energy demand']
    0.1939229 ['public transport', 'public roadways', 'public roads', 'public bus', 'public transit services']
    0.18963552 ['Many experts', 'experts', 'Expertswho', 'respondents', 'opinion surveys']
    0.18766627 ['effective', 'usable', 'accurate', 'usefulness', 'desirable']
    0.16964775 ['certain industry standards', 'certain standardized interfaces', 'standards', 'performance standards', 'certain conditions']
    0.16584705 ['frequency', 'noise', 'active', 'air', 'sky']
    0.16010725 ['assessment', 'look', 'features', 'characteristics', 'details']
    0.15609962 ['highway capacity', 'highway', 'highway infrastructure', 'highway speeds', 'roadway capacity']
    0.1467838 ['technological development', 'technology', 'technological progress', 'innovation', 'innovation strategy development']
    0.1437942 ['users', 'customers', 'consumers', 'people', 'participants']
    0.13379529 ['complete', 'substantial actual use', 'complete implementation', 'real conditions', 'entirety']
    0.11945792 ['criminal liability', 'liability', 'legal liability', 'liable', 'liability laws']
    0.10100062 ['overall number', 'total', 'requirement', 'duty', 'task']
    0.10004056 ['similar', 'similar approaches', 'similar things', 'equivalent', 'related']
    0.09786007 ['German drivers', 'Mercedes-Benz', 'Mercedes-Benz S-Class', 'Daimler', 'German population']
    0.09186755 ['opportunity', 'potential', 'idea', 'attempt', 'scenario']
    0.084540494 ['areas', 'extent', 'land', 'hectares', 'settlement']
    0.08340411 ['likely', 'probable outcomes', 'foreseeability', 'plausible', 'suitable']
    0.08213691 ['US states', 'US government state', 'USA', 'United States', 'states']
    0.081921406 ['commercial', 'advertising', 'market segment', 'marketing', 'retail commerce']
    0.072908044 ['decrease', 'less attention', 'lower intention', 'less need', 'least']
    0.06899645 ['ALKS', 'STRIA', 'level ALKS', 'UVEK', 'ALV']
    0.06597975 ['Delphi technology', 'Delphi', 'Delft University', 'Zalaegerszeg', 'Netherlands']
    0.061130047 ['personnel', 'workers', 'workforce', 'jobs', 'work']
    0.060929984 ['European Union', 'United Kingdom', 'European Commission', 'Europe', 'Great-Britain']
    0.05525419 ['unmatched', 'nonexistent', 'undecided', 'unprepared', 'unquestionable failure']
    0.05455757 ['Nevada Department', 'Nevada', 'California', 'California Department', 'Nevada law']
    0.05267317 ['SAE standard', 'SAE', 'SAE level', 'formal SAE definition', 'SAE International']
    0.051717993 ['important', 'significant impact', 'notable effect', 'major implications', 'essential']
    0.048221324 ['legislation', 'specific laws', 'law', 'Policy-makers', 'revising laws']
    0.04552625 ['problem', 'difficulty', 'concerns', 'significant uncertainties', 'confusion']
    0.04302664 ['classification system', 'classification', 'particular', 'specificity', 'District']
    0.04061676 ['example', 'instance', 'term', 'meaning', 'definition']
    0.039021783 ['cooperation', 'connected', 'partnership', 'interconnectivity', 'alliance']
    0.0375225 ['different sorts', 'different types', 'different systems', 'different sources', 'different opinions']
    0.03453212 ['Baidu', 'CMU', 'Hubei province', 'Hubei China', 'Udacity']
    0.02412247 ['first occurrence', 'beginning', 'first', 'emergence', 'early']
    0.021739865 ['large amounts', 'large numbers', 'larger scale', 'extensive amount', 'large']
    0.004845136 ['new', 'new technology', 'new look', 'new system', 'new kind']
    -0.0041739354 ['fatal accident', 'fatal situation', 'fatality', 'fatal crashes', 'similar fatal crash']
    -0.007475814 ['changes', 'transition', 'turn', 'shift', 'transition phase']
    -0.028057314 ['increase', 'exponential increase', 'fast increase', 'necessary increase', 'incremental advances']
    -0.0586837 ['years old', 'years', 'age', 'old', 'decades']
    -0.13896333 ['month', 'March', 'March update', 'early October', 'February']



```python
# Sort by similarity to original document
wsr_d = centr_sort(words, word_emb, lbs, doc_emb, sort_by="doc")

print("\n\nSorted by original doc\n")
for w in wsr_d:
    print(w[0], w[1][:5])
```

    
    
    Sorted by original doc
    
    0.62424296 ['autonomous vehicles market', 'autonomous vehicle', 'autonomous car', 'autonomous vehicle industry', 'many autonomous vehicles']
    0.5434958 ['self-driving car project', 'Modern self-driving cars', 'self-driving-car testing', 'self-driving car industry', 'self-driving car story']
    0.4824011 ['ultrasonic sensors', 'sensory data', 'sensory information', 'electronic blind-spot assistance', 'adaptive cruise control']
    0.4445539 ['vehicle control method', 'vehicle communication systems', 'vehicle transmitting data', 'vehicle control', 'vehicle-to-vehicle communication']
    0.44330907 ['driver assistance technologies', 'driver- assistance systems', 'advanced driver-assistance systems', 'advanced driver assistance systems', 'Driving Systems']
    0.4168215 ['car navigation system', 'automotive vehicles', 'automobiles', 'computer-controlled car', 'car technology']
    0.39603418 ['vehicle automation', 'semi-automated vehicles', 'first semi-automated car', 'Automated cars', 'automated vehicles']
    0.38521442 ['motion sensors', 'surround cameras', 'sensors', 'Typical sensors', 'sensor systems']
    0.3617182 ['hybrid navigation', 'navigation system', 'GPS', 'GPS position', 'appropriate navigation paths']
    0.34224677 ['robotic car', 'Starsky Robotics', 'robotic vehicles', 'Robotics', 'robot workers']
    0.34062952 ['driverless systems', 'driverless technology', 'Driverless vehicles', 'driverless car', 'first electric driverless racing car']
    0.33944237 ['self-protective cars', 'driver safety', 'safety driver', 'passenger privacy', 'Traffic Safety']
    0.33884573 ['Autopilot technology', 'Autopilot mode', 'automatic driving', 'manual driving', 'autopilot system']
    0.3067453 ['individual cars', 'human driver input', 'individual vehicle', 'single drivers', 'total private vehicle use']
    0.27357867 ['artificial intelligence', 'probabilistic machine learning vehicle AI', 'Artificial Intelligence Laboratory', 'artificial aids', 'software hacking']
    0.26621902 ['roadside real-time locating system', 'location-based ads', 'location system', 'remote places', 'surroundings']
    0.26551753 ['Vehicular communication systems', 'recorder', 'voice recording', 'voiceswho', 'communication systems']
    0.25277805 ['auto-piloted car', 'human driver control', 'AutonoDrive', 'robo-car', 'Tesla Autopilot capability']
    0.25201482 ['vehicle energy efficiency', 'fuel efficiency improvement', 'fuel economy', 'fuel savings', 'engine']
    0.23806205 ['system operator', 'operating principles', 'operating software', 'software systems', 'operating system']
    0.23448318 ['utilitarian vehicles', 'utilitarian ethics', 'utilitarian views', 'utilitarian ideas', 'utilitarianism']
    0.20482199 ['traffic-control devices', 'lane keeping systems', 'traffic police', 'real-time traffic information', 'traffic lights']
    0.20016664 ['interior', 'housing affordability', 'roadside restaurants', 'interior design industry', 'deep learning architecture']
    0.19620562 ['battery-powered', 'air pollution', 'batteries', 'energy', 'electricity']
    0.1939229 ['trolley', 'transport systems', 'Transportation', 'public road tests', 'public transit services']
    0.18963552 ['Expertswho', 'human-factor challenges', 'experts', 'respondents', 'Many experts']
    0.18766627 ['self-interested preferences', 'autonomy', 'accurate detection', 'improved convenience', 'partial autonomy']
    0.16964775 ['automotive standardization body', 'certain characteristics', 'certain modules', 'certain industry standards', 'certain aspects']
    0.16584705 ['active control', 'explosives', 'noise', 'air', 'gunfire']
    0.16010725 ['characteristics', 'verification', 'investigative body', 'perspective', 'assessment']
    0.15609962 ['Florida Highway Patrol', 'unmapped roads', 'non-controlled access highway', 'US National Transportation Safety Board', 'limited-access highways']
    0.1467838 ['smart systems', 'smart machine', 'accompanying technologies', 'smart technological advances', 'technology']
    0.1437942 ['Consumer Electronic Show visitors', 'consumer expectation test', 'potential consumer opinion', 'ambiguous user preferences', 'occupants']
    0.13379529 ['real-world conditions', 'comprehensive product experience', 'outright ownership', 'entirety', 'entire job categories']
    0.11945792 ['liability laws', 'normal liability', 'criminals', 'product liability', 'liability']
    0.10100062 ['average speed', 'average distance', 'provider', 'service', 'aggregate']
    0.10004056 ['behavior predictability', 'real-time maps', 'behavior', 'behavioral patterns', 'map matching']
    0.09786007 ['PSA Peugeot-Citroen', 'Audi', 'Mercedes premi', 'Mercedes-Benz', 'Mercedes-Benz S-Class']
    0.09186755 ['maneuver', 'side-swiped', 'direction', 'open road', 'concepts leads']
    0.084540494 ['areas', 'land', 'extent', 'earth', 'world']
    0.08340411 ['expectation', 'likelihood', 'plausible', 'presumed', 'strong likelihood']
    0.08213691 ['many Americans', 'Virginia', 'USA', 'Governor Jerry Brown', 'America']
    0.081921406 ['LED signs', 'advertisement business', 'relevant signage', 'commercial projects', 'business opportunities']
    0.072908044 ['non-verbal cues', 'non-automated', 'non-blind person', 'Mundane ethical situations', 'Mundane situations']
    0.06899645 ['ALV projects', 'ALKS', 'UVEK', 'magnetic strips', 'ALV']
    0.06597975 ['Delphi technology', 'Netherlands', 'Zalaegerszeg', 'South Korea', 'South Korean government']
    0.061130047 ['human operator', 'crash repair shops', 'dogs', 'repair industry', 'conductor']
    0.060929984 ['third parties', 'uninvolved third party', 'Spain', 'Europe', 'EuroNCAP']
    0.05525419 ['unresolved questions', 'untestable potential', 'unmatched', 'unconnected one', 'undecided']
    0.05455757 ['Arizona Governor Doug Ducey', 'Arizona State University', 'Arizona', 'California Department', 'California Assembly Bill']
    0.05267317 ['Washington', 'formal SAE definition', 'national territory', 'SAE', 'SAE International']
    0.051717993 ['interests', 'highlights', 'factors', 'parties responsible', 'important reason']
    0.048221324 ['Appropriate public policies', 'state legislation', 'state laws', 'legal status', 'policy intervention']
    0.04552625 ['evasive actions', 'random scenarios', 'unlikely fact patterns', 'malicious interveners', 'obstacles']
    0.04302664 ['Taxonomy', 'type', 'district code', 'boundaries', 'code']
    0.04061676 ['AAA Foundation', 'AAA', 'gestures', 'context', 'Association']
    0.039021783 ['cooperative networking', 'cooperative approach', 'interconnectivity', 'close attention', 'cooperation']
    0.0375225 ['different vehicles', 'seizures', 'different types', 'different techniques', 'different sources']
    0.03453212 ['CAV', 'Mohan', 'Sripad', 'Hubei China', 'Hubei province']
    0.02412247 ['access', 'emergence', 'Asimov', 'tailgating', 'approach']
    0.021739865 ['enhanced processing capabilities', 'several meaning', 'several modules', 'wide access', 'full speed']
    0.004845136 ['newer vehicles technology name', 'new navigation maps', 'new transportation technology', 'new energy ecosystem', 'new technology']
    -0.0041739354 ['involved rear-end collisions', 'crash rate', 'crashes', 'collisions', 'fatal crashes']
    -0.007475814 ['turn signals', 'transformative', 'decoupling', 'forefront', 'design']
    -0.028057314 ['extra data', 'elevated rail', 'incremental permutations', 'incremental advances', 'rapid advance']
    -0.0586837 ['era', 'US adults', 'elderly', 'lifetimes', 'old']
    -0.13896333 ['updates', 'NHTSA', 'May', 'Kazan', 'updates schedule']


### Granularity of clusters 
Granularity of clusters could be controled by decreasing *min_cluster_size*


```python
# Decrease 'min_cluster_size' to get more fine topics
lbs = get_clusters(word_emb, min_cluster_size=6)
```


```python
# Sort by similarity to centroind of cluster
wsr_c = centr_sort(words, word_emb, lbs, doc_emb)

print("\n\nSorted by centroids\n")
for w in wsr_c:
    print(w[0], w[1][:5])
```

    
    
    Sorted by centroids
    
    0.61354375 ['autonomous car', 'autonomous driving', 'landmark autonomous car', 'autonomous vehicles market', 'autonomous vehicle industry']
    0.5766833 ['autonomous system', 'autonomous mode', 'autonomous transportation', 'autonomous', 'autonomous capacity']
    0.5384934 ['self-driving car', 'self-driving', 'self-driving car industry', 'full self-driving car', 'self-driving car project']
    0.4755486 ['visual perception', 'radar perception', 'perception', 'visual object recognition', 'computer vision']
    0.46143538 ['driver- assistance systems', 'driver assistance technologies', 'advanced driver-assistance systems', 'advanced driver assistance systems', 'driver intervention']
    0.45160204 ['automotive vehicles', 'automobiles', 'automotive', 'automotive industry', 'vehicles']
    0.41427082 ['self-parking technology', 'self-parking systems', 'self-parking', 'parking', 'parking space']
    0.4000696 ['automated vehicles', 'Automated cars', 'automated features', 'Automated Transport', 'semi-automated vehicles']
    0.39545935 ['driver', 'driver position', 'driver seat', 'driver experience', 'Driver interactions']
    0.39179915 ['Robotics', 'robotic vehicles', 'robotic car', 'robot workers', 'Starsky Robotics']
    0.38493598 ['sensors', 'sensor systems', 'sensor-based infrastructure', 'Typical sensors', 'motion sensors']
    0.3717926 ['navigation system', 'GPS', 'GPS position', 'appropriate navigation paths', 'satellite navigation system']
    0.36927554 ['driving task', 'Driving Systems', 'driving', 'driving features', 'steering wheel']
    0.3673584 ['automation technology', 'automation level', 'automation', 'automation systems', 'automation level definitions']
    0.36102825 ['driverless car', 'driverless', 'Driverless vehicles', 'driverless technology', 'driverless systems']
    0.35818917 ['test vehicles', 'vehicle research', 'vehicle technology', 'vehicle concepts', 'development vehicles']
    0.35225344 ['cameras', 'video recording', 'video', 'surround cameras', 'backup camera']
    0.3481965 ['devices', 'electronic devices', 'mobile device', 'inertial measurement units', 'electronic map']
    0.33946657 ['safety', 'public safety', 'safety hazards', 'safety issues', 'protection']
    0.3210839 ['brake pedal', 'brakes', 'emergency braking', 'brake lights', 'Autonomous Emergency Braking']
    0.30802843 ['artificial intelligence', 'Artificial Intelligence Laboratory', 'software hacking', 'machine learning', 'hacking']
    0.2992523 ['autopilot system', 'autopilot', 'Autopilot mode', 'Autopilot technology', 'autopilot system error']
    0.29309916 ['ethical issues', 'specific ethical frameworks', 'ethical problems', 'ethical preferences', 'ethical theories']
    0.28851366 ['Tesla Motors', 'Tesla Autopilot', 'Tesla', 'Tesla cars', 'Tesla Autopilot capability']
    0.28401572 ['location', 'vicinity', 'surroundings', 'location system', 'geographic traits']
    0.2816658 ['surveillance', 'surveillance apparatus', 'mass surveillance', 'monitoring', 'tracking']
    0.26924506 ['insurance industry', 'automobile insurance industry', 'auto insurance industry', 'car insurance', 'insurances']
    0.26836517 ['license', 'permit', 'exemption', 'licensed drivers', 'licensed operator']
    0.2612735 ['software systems', 'software', 'operating software', 'operating system', 'software code']
    0.2562544 ['mode', 'components', 'units', 'equipped', 'individual']
    0.25552905 ['fuel efficiency improvement', 'higher fuel efficiency', 'fuel economy', 'fuel savings', 'higher fuel economy']
    0.2546237 ['human', 'human occupants', 'human actions', 'human drivers', 'human life']
    0.2476248 ['communication', 'communication systems', 'communication connection', 'communication networks', 'Vehicular communication systems']
    0.22774497 ['mobility', 'mobility-impaired', 'mobility providers', 'greater mobility', 'enhanced mobility']
    0.22771011 ['information', 'data processing', 'information available', 'processing', 'reprogrammable characteristics']
    0.22472511 ['human-error', 'human error', 'human failures', 'human-factor challenges', 'human intervention']
    0.2090881 ['Road Traffic', 'traffic', 'traffic collisions', 'traffic congestion', 'traffic-jam']
    0.19865811 ['assessment', 'investigative body', 'investigation', 'look', 'agency']
    0.1970978 ['digital information', 'digital characteristics', 'digital technology', 'digital era', 'digital traces']
    0.19552967 ['entertainment-', 'recreational areas', 'entertainment industry', 'leisure', 'media-entertainment industry']
    0.18522114 ['personal ownership', 'private property', 'owner', 'personal preferences', 'current owners']
    0.18454894 ['Mercedes-Benz', 'Mercedes premi', 'Mercedes-Benz S-Class', 'Mercedes-Benz ML Matic', 'PSA Peugeot-Citroen']
    0.18391876 ['public transport', 'public roadways', 'public bus', 'public transit services', 'public roads']
    0.18078181 ['Google Corporation', 'Google', 'Google vehicle', 'Google Headquarters', 'Michigan Department']
    0.17566109 ['customers', 'consumers', 'users', 'consumer behavior', 'potential consumer opinion']
    0.17140967 ['Many industries', 'Many experts', 'experts', 'industry', 'Many historical projects']
    0.1697676 ['buildings', 'structure', 'housing', 'architecture', 'interior design']
    0.16838586 ['technological progress', 'technological development', 'technology', 'breakthrough technological advances', 'tech']
    0.16822976 ['certain aspects', 'certain updates', 'certain conditions', 'certain characteristics', 'certain replaceable parts']
    0.16786161 ['energy consumption', 'electricity consumption', 'energy', 'electricity', 'emissions']
    0.16640079 ['effective', 'capable', 'usable', 'suitable', 'accurate']
    0.16103306 ['Computer Science', 'computer programs', 'computers', 'computer scientists', 'computer simulations']
    0.15978199 ['advertising', 'ADS', 'advertisement business', 'marketing', 'commercial']
    0.15959175 ['noise', 'bombs', 'explosives', 'frequency', 'gunfire']
    0.15941215 ['easier', 'safer', 'ease', 'comfortable', 'safe']
    0.15704283 ['precise', 'proper', 'confident', 'aware', 'applicable']
    0.1488674 ['trucks', 'long-distance trucking', 'heavy vehicles', 'Large vehicles', 'tractor-trailer']
    0.14458281 ['available road space', 'roads', 'road work', 'road space due', 'Roadmap']
    0.14253809 ['highway infrastructure', 'highway', 'roadway capacity', 'highway capacity', 'intersections']
    0.14173648 ['cultures', 'content creators', 'website', 'countries', 'online survey']
    0.13722569 ['urban areas', 'urban', 'urban travel', 'urban design', 'city']
    0.1354626 ['questionnaire survey', 'survey', 'respondents', 'opinion surveys', 'US telephone survey']
    0.13211162 ['usefulness', 'beneficial', 'benefits', 'desirable', 'reward']
    0.1312653 ['New York state', 'Delft University', 'Delphi technology', 'NYS DMV', 'New York State officials']
    0.124120235 ['complete', 'substantial actual use', 'complete implementation', 'real conditions', 'whole']
    0.12082396 ['travel distances', 'travel', 'travel time', 'routes', 'kilometers']
    0.11841071 ['standards', 'performance standards', 'industry standards', 'international standards', 'technical specifications']
    0.114548825 ['moral problem', 'moral basis', 'moral dilemma', 'moral responsibility', 'Moral Machine']
    0.1109826 ['assistance', 'forgiveness', 'reparation', 'care', 'risk compensation']
    0.105085075 ['research', 'research organizations', 'scientists', 'Experiments', 'Concerned Scientists']
    0.10466656 ['way', 'manner', 'Thatcham', 'sort', 'direction']
    0.10319631 ['personnel', 'Military personnel', 'Men', 'general', 'human operator']
    0.10227963 ['innovation', 'Innovation Agenda', 'innovation strategy development', 'disruptive innovation', 'stifle innovation']
    0.099148065 ['criminal liability', 'strict liability', 'legal liability', 'liability', 'liability laws']
    0.09737246 ['CAV', 'Udacity', 'ALV', 'Vaishnav', 'level ALKS']
    0.09102784 ['task', 'operation', 'technique', 'role', 'maneuver']
    0.08804639 ['US states', 'USA', 'US government state', 'United States', 'states']
    0.085041896 ['pluralism', 'localization', 'homogenization', 'charge', 'position']
    0.0814565 ['Nevada', 'Nevada Department', 'Nevada law', 'Nevada Legislature', 'Nevada Supreme Court ruling']
    0.07995793 ['law', 'legal framework', 'legislation', 'legal status', 'legal obligations']
    0.07828452 ['cooperation', 'connected', 'partnership', 'interconnectivity', 'alliance']
    0.07613675 ['snowy roads', 'snow', 'Cloud', 'Yorkshire', 'Nordic Communications Corporation']
    0.07550965 ['decrease', 'absence', 'decline', 'end', 'lack']
    0.07527004 ['specific requirements', 'requirement', 'specific list', 'specific set', 'task requirements']
    0.07447592 ['money', 'financial', 'payouts', 'investment', 'funding']
    0.0732031 ['Hubei China', 'Hubei province', 'Zhengzhou', 'China', 'Henan province']
    0.07287039 ['similar', 'similar approaches', 'similar things', 'equivalent', 'map matching']
    0.072815 ['widespread use', 'widespread acceptance', 'widespread adoption', 'large scale implementation', 'popularity']
    0.07265402 ['test-driven', 'testing', 'trial run', 'trials', 'public testing']
    0.07054664 ['situations', 'circumstances', 'involved', 'factors', 'scenario']
    0.06865168 ['European Commission', 'Europe', 'European Union', 'third', 'European Parliament']
    0.06862442 ['technical failure', 'technical malfunction', 'technical issue', 'technical default', 'design defect']
    0.062727444 ['possibilities', 'potential', 'options', 'opportunity', 'open possibilities']
    0.061378337 ['overall number', 'total', 'amount', 'capacity', 'size']
    0.061118253 ['summer', 'August', 'July', 'September', 'June']
    0.059244722 ['different types', 'different systems', 'different sorts', 'different sources', 'different techniques']
    0.05525419 ['unmatched', 'nonexistent', 'undecided', 'unprepared', 'unquestionable failure']
    0.054361537 ['hectares', 'extent', 'areas', 'acres', 'Valeo']
    0.05267317 ['SAE standard', 'SAE', 'SAE level', 'formal SAE definition', 'SAE International']
    0.052054856 ['problem', 'difficulty', 'concerns', 'significant uncertainties', 'confusion']
    0.051277958 ['classification system', 'classification', 'District', 'district code', 'formal classification system']
    0.045518346 ['incremental change', 'changes', 'transition', 'incremental shifts', 'transition phase']
    0.044623896 ['first occurrence', 'beginning', 'first', 'emergence', 'introduction']
    0.043696065 ['highest', 'greatest good', 'maximum', 'record achievement', 'maximum degree possible']
    0.042934086 ['significant impact', 'major implications', 'important', 'notable effect', 'tremendous impact']
    0.04274254 ['lower levels', 'lower intention', 'less attention', 'less need', 'less money']
    0.040225692 ['policy reform', 'Policy-makers', 'policy', 'policymakers', 'policy intervention']
    0.040113214 ['education', 'school industry', 'higher education', 'schools', 'University']
    0.040023938 ['California Department', 'California', 'Japan', 'San Diego', 'Honda']
    0.039956786 ['combination', 'combined effect', 'variant', 'double function', 'control possible']
    0.037007023 ['retail commerce', 'market demands', 'market segment', 'hypermarkets', 'mass market']
    0.033402763 ['workers', 'workforce', 'jobs', 'work', 'labor costs']
    0.030814767 ['United Kingdom', 'Great-Britain', 'British English', 'British legal definition', 'British law']
    0.02808682 ['example', 'term', 'instance', 'meaning', 'definition']
    0.025738668 ['large amounts', 'large numbers', 'extensive amount', 'large', 'wide scale']
    0.017541528 ['higher speeds', 'larger scale', 'greater proportion', 'greater emphasis', 'higher levels']
    0.016260844 ['people', 'person', 'population', 'participants', 'gender']
    0.01580438 ['decision process', 'discussions', 'decisions', 'opinion', 'interviewees']
    0.012252424 ['France', 'French government', 'French companies', 'Paris area', 'Île-de-France']
    0.002313245 ['German population', 'Germany', 'German law', 'German drivers', 'Otto']
    -0.009471171 ['Baidu', 'IMU', 'CSAIL', 'Hulu', 'Zalaegerszeg']
    -0.013644533 ['new', 'new system', 'new technology', 'new look', 'new entrants']
    -0.031472545 ['increase', 'exponential increase', 'necessary increase', 'fast increase', 'rise']
    -0.0457953 ['fatal accident', 'fatal situation', 'fatality', 'first fatal accident', 'fatal crashes']
    -0.0586837 ['years old', 'years', 'age', 'old', 'decades']
    -0.073773146 ['young', 'Children', 'teens', 'young passengers', 'young men']
    -0.09085511 ['March', 'April', 'February', 'March update', 'May']
    -0.09958036 ['second challenge', 'second occurrence', 'second state', 'latter term', 'replacement']
    -0.17684199 ['month', 'November', 'early October', 'November update', 'October']




