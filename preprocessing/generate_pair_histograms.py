#
# Histogram generator, for the corpus docs <-> pre-ranked query results
#   pre-rank gives us pairs of docs <-> topics
#
# Input:  ../data/trec_corpus.txt, topics file, pre-ranked 2k file, gensim word embedding model, bin count
# Output: (fixed) ../data/trec_corpus__histogram_<bin count>.txt
#         contents: topicId docId 1 2 3 8 0 2 3 ...
#                   topicId docId 1 2 3 8 0 2 3 ...
#
# The pre-rank has to be created with the topic file + stemmed, stopword list processed documents referenced here !!
#

import os
import sys
from gensim.models import Word2Vec
import numpy as np
import timeit

start_time = timeit.default_timer()

# from matchzoo: https://github.com/faneshion/MatchZoo/blob/master/matchzoo/inputs/preprocess.py
def cal_hist(t1_rep, t2_rep, qnum, hist_size):

    mhist = np.zeros((qnum, hist_size), dtype=np.float32)
    mm = t1_rep.dot(np.transpose(t2_rep))

    for (i,j), v in np.ndenumerate(mm):
        if i >= qnum:
            break
        vid = int((v + 1.) / 2. * (hist_size - 1.))
        mhist[i][vid] += 1.

    mhist += 1.
    mhist = np.log10(mhist)
    return mhist.flatten()


# make sure the argument is good (0 = the python file, 1 the actual argument)
if len(sys.argv) < 6:
    print 'Needs 5 arguments - see comments for info ...'
    exit(0)

arg_corpus_file = sys.argv[1]
arg_topics_file = sys.argv[2]
arg_preranked_file = sys.argv[3]
arg_embedding_file = sys.argv[4]
arg_bin_size = int(sys.argv[5])

# load word embedding
model = Word2Vec.load(arg_embedding_file)
vectors = model.wv
del model
vectors.init_sims(True) # normalize the vectors (!), so we can use the dot product as similarity measure

print 'embeddings loaded '
print 'loading docs ... '

# load trec corpus
trec_corpus={} # corpus id -> list of doc vector ids
count = 0
with open(arg_corpus_file, 'r') as inputFile:
    for line in inputFile:
        count+=1
        if count % 10000==0:
            print '    ', count,' docs loaded'
        parts = line.split(' ', 1)
        trec_corpus[parts[0]] = []
        for w in parts[1].split(' '):
            ws = w.strip()
            if ws in vectors.vocab:
                trec_corpus[parts[0]].append(vectors.vocab[ws].index)

print 'all ', count, ' docs loaded'

# load topics file
trec_topics = {} # topic -> list of query term vector ids
with open(arg_topics_file, 'r') as inputFile:
    for line in inputFile:
        parts = line.split(' ', 1)

        if parts[0] not in trec_topics:
            trec_topics[parts[0]] = []

        for w in parts[1].split(' '):
            ws = w.strip()
            if ws in vectors.vocab:
                trec_topics[parts[0]].append(vectors.vocab[ws].index)

print 'all ', len(trec_topics), ' topics loaded'

# load pre-ranked file
pre_ranked_per_topic = [] # topic -> (doc id, rank)
with open(arg_preranked_file, 'r') as inputFile:
    for line in inputFile:
        parts = line.split(' \t ') # pretty strange, but it works
        pre_ranked_per_topic.append((parts[0], parts[2].strip(), parts[3].strip()))

print 'all ', len(pre_ranked_per_topic), ' pre-ranked topics loaded'

print 'creating histograms'
count = 0
# create histograms for every query term <-> doc term
# based on pairs from pre-ranked file, using the similarities of the wordembedding
with open('../data/topic_corpus_histogram_'+str(arg_bin_size)+'.txt', 'w') as outputFile:

    for topic, doc, rank in pre_ranked_per_topic:
            count += 1
            if count % 10000 == 0:
                print '    ', count, ' ranked docs processed'

            # get the word embedding ids
            topic_word_ids = trec_topics[topic]
            doc_words_ids = trec_corpus[doc]

            topic_vectors = np.array([vectors.word_vec(vectors.index2word[topic_word_ids[i]], True) for i in range(0,len(topic_word_ids))],np.float32)
            doc_vectors = np.array([vectors.word_vec(vectors.index2word[doc_words_ids[i]], True) for i in range(0,len(doc_words_ids))],np.float32)

            outputFile.write(topic+" "+doc+" "+str(len(topic_word_ids))+" ")

            qnum = len(topic_word_ids)
            d1_embed = topic_vectors
            d2_embed = doc_vectors

            curr_hist = cal_hist(d1_embed, d2_embed, qnum, arg_bin_size)
            curr_hist = curr_hist.tolist()
            outputFile.write(' '.join(map(str, curr_hist)))

            outputFile.write('\n')
            outputFile.flush()

print 'Completed after (seconds): ', timeit.default_timer() - start_time
