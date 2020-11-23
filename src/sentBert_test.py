from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
from gensim.parsing.preprocessing import strip_multiple_whitespaces, strip_non_alphanum, strip_tags
import re
import pickle


def clean(sx):
    sx = strip_tags(sx)
    sx = strip_non_alphanum(sx)
    sx = re.sub(r'\n',' ',sx)
    sx = strip_multiple_whitespaces(sx)
    return sx

enty = pd.read_table('../data/concepts.tsv', names=['link','name','type','description'],header=0)
enty['clean_text']= enty["description"].apply(clean)
enty.drop_duplicates(subset=["name"], keep='first', inplace=True)


embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')


# Corpus with example sentences
corpus  = enty['clean_text'].tolist()


# ~ emb = embedder.encode(corpus, convert_to_tensor=True)
# ~ #Store sentences & embeddings on disc
# ~ with open('../out/embeddings.pkl', "wb") as fOut:
    # ~ pickle.dump({'corpus': corpus, 'embeddings': emb}, fOut, protocol=pickle.HIGHEST_PROTOCOL)


intemb = pickle.load(open('../out/embeddings.pkl', "rb"))

emb = intemb['embeddings']

# Query sentences:
queries = ["A self-driving car, also known as an autonomous vehicle (AV), connected and autonomous vehicle (CAV), full self-driving car or driverless car, or robo-car or robotic car, (automated vehicles and fully automated vehicles in the European Union) is a vehicle that is capable of sensing its environment and moving safely with little or no human input.",
        "Self-driving cars combine a variety of sensors to perceive their surroundings, such as radar, lidar, sonar, GPS, odometry and inertial measurement units. Advanced control systems interpret sensory information to identify appropriate navigation paths, as well as obstacles and relevant signage.",
        "Connected vehicle platoons and long-distance trucking are seen as being at the forefront of adopting and implementing the technology."
            ]


# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = 5

def get_concept(qr,emb,enty_df,topk=5):
    qr_em = embedder.encode(qr, convert_to_tensor=True)
    cos_sc = util.pytorch_cos_sim(qr_em, emb)[0]
    cos_sc = cos_sc.cpu()

    #We use torch.topk to find the highest 5 scores
    top_K = torch.topk(cos_sc, k=topk)

    print("\n\n======================\n\n")
    print("Query:", qr)
    print("\nTop  most similar concepts in corpus:")

    for score, idx in zip(top_K[0], top_K[1]):
        print(enty_df.iloc[int(idx)]['name'],'/',enty_df.iloc[int(idx)]['type'], "(Score: %.4f)" % (score))
    
for query in queries:
    get_concept(query,emb,enty,topk=15)














# ~ import spacy
# ~ nlp = spacy.load("en")
# ~ text = "But Google is starting from behind. The company made a late push\ninto hardware, and Apple’s Siri, available on iPhones, and Amazon’s Alexa\nsoftware, which runs on its Echo and Dot devices, have clear leads in\nconsumer adoption."
# ~ doc = nlp(text)
# ~ for ent in doc.ents:
    # ~ print(ent.text, ent.start_char, ent.end_char, ent.label_)

















# ~ model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

# ~ #Our sentences we like to encode
# ~ sentences = ['This framework generates embeddings for each input sentence',
    # ~ 'Sentences are passed as a list of string.',
    # ~ 'The quick brown fox jumps over the lazy dog.']

#Sentences are encoded by calling model.encode()
# ~ embeddings = model.encode(sentences)

#Print the embeddings
# ~ for sentence, embedding in zip(sentences, embeddings):
    # ~ print("Sentence:", sentence)
    # ~ print("Embedding:", embedding)
    # ~ print("")

#Sentences are encoded by calling model.encode()
# ~ emb1 = model.encode("This is a red cat with a hat.")
# ~ emb2 = model.encode("Have you seen my red cat?")

# ~ cos_sim = util.pytorch_cos_sim(emb1, emb2)
# ~ print("Cosine-Similarity:", cos_sim)

# ~ model = SentenceTransformer('distilroberta-base-msmarco-v2')
# ~ query_embedding = model.encode('How big is London')
# ~ passage_embedding = model.encode('London has 9,787,426 inhabitants at the 2011 census')

# ~ print("Similarity:", util.pytorch_cos_sim(query_embedding, passage_embedding))
