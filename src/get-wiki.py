from bs4 import BeautifulSoup
import requests
from gensim.parsing.preprocessing import strip_multiple_whitespaces, strip_non_alphanum, strip_tags, strip_numeric
import re



# ==  Functions
def clean(sx):
    sx = strip_tags(sx)
    sx = strip_numeric(sx)
    sx = re.sub(r'\n',' ',sx)
    sx = re.sub(r'\[','',sx)
    sx = re.sub(r'\]','',sx)
    sx = strip_multiple_whitespaces(sx)
    return sx
# ===========================================


fout = '../data/self-car.txt'
link = 'https://en.wikipedia.org/wiki/Self-driving_car'
res = requests.get(link)

if res is not None:
    html = BeautifulSoup(res.text, 'html.parser')
    pars = html.select('p')
    
    with open(fout, 'w') as fo:
        for par in pars:
            # ~ fo.write(" ".join(simple_preprocess(par.text, deacc=True))+'\n')
            fo.write(clean(par.text)+'\n')
