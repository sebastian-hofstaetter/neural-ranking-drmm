#
# Finds the out of vocabulary words for the given trec corpus/ trec titles file and the given gensim wordembedding
#
# Input:  Stemmed word embedding model (porter stemmer)
#         Non-stemmed, but text only (id text) file - words will be stemmed with nltk porter stemmer

# Output: console prints out of vocab words in a sorted set fashion
#

from gensim.models import Word2Vec
import sys
import os
from nltk.stem import *
from nltk.corpus import stopwords

# make sure the argument is good (0 = the python file, 1,2 the actual argument)
if len(sys.argv) != 3 or not os.path.isfile(sys.argv[1]) or not os.path.isfile(sys.argv[2]):
    print 'Needs 2 arguments - 1. word2vec gensim model, 2. the trec file (id text) that you want to check'
    exit(0)

#
# load word2vec model
#
model = Word2Vec.load(sys.argv[1])
vectors = model.wv
del model

#
# check vocab
#
stemmer = PorterStemmer()
checked = 0
stopWords = set(stopwords.words('english'))

with open(sys.argv[2], 'r') as inFile:
    wordlist = []
    #i = 0
    for line in inFile:
        #i += 1
        #if i > 1000:
        #    break
        checked += 1
        parts = line.split(' ',1)
        for word in parts[1].split(' '):
            ws = word.strip()
            if ws not in stopWords:
                stemmed = stemmer.stem(ws)
                if stemmed not in vectors.vocab:
                    wordlist.append((ws, stemmed))

print 'found ', len(wordlist),' oov words'
if len(wordlist) > 0:

    x = [a[0]+' ('+a[1]+')' for a in set(wordlist)]
    x.sort()
    print '\n'.join(x)

print '\n-------\n', checked, ' lines checked'
