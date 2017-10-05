#
# Histogram generator, for the corpus docs <-> pre-ranked query results
#   pre-rank gives us pairs of docs <-> topics
#
# Input:  ../data/trec_corpus.txt, topics file, pre-ranked 2k file, gensim word embedding model, bin count, 'qrels' or 'prerank'
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
from TextCollection import TextCollection

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


# make sure the argument is good (0 = the python file, 1+ the actual argument)
if len(sys.argv) < 7:
    print('Needs 6 arguments - see comments for info ...')
    exit(0)

arg_corpus_file = sys.argv[1]
arg_topics_file = sys.argv[2]
arg_preranked_or_qrel_file = sys.argv[3]
arg_embedding_file = sys.argv[4]
arg_bin_size = int(sys.argv[5])
arg_qrel_or_preranked = sys.argv[6] # qrel or prerank

#
# load pre-ranked or qrels file
#
topic_doc_pairs = [] # (topic, doc id, score) score is 0,1 if qrel

if arg_qrel_or_preranked == 'prerank':
    with open(arg_preranked_or_qrel_file, 'r') as inputFile:
        for line in inputFile:
            parts = line.split()
            topic_doc_pairs.append((parts[0], parts[2].strip(), parts[4].strip()))

if arg_qrel_or_preranked == 'qrel':
    with open(arg_preranked_or_qrel_file, 'r') as inputFile:
        for line in inputFile:
            parts = line.split()
            topic_doc_pairs.append((parts[0], parts[2].strip(), parts[3].strip()))

print('all ', len(topic_doc_pairs), ' topic_doc pairs loaded')


# load word embedding
model = Word2Vec.load(arg_embedding_file)
vectors = model.wv
del model
vectors.init_sims(True) # normalize the vectors (!), so we can use the dot product as similarity measure

print('embeddings loaded ')
print('loading docs ... ')

# load trec corpus
trec_text_collection_data = [] # text 1 string per doc only, no id
trec_corpus={} # corpus id -> list of doc vector ids
count = 0
with open(arg_corpus_file, 'r') as inputFile:
    for line in inputFile:
        count+=1
        if count % 10000==0:
            print('    ', count,' docs loaded')
        parts = line.split(' ', 1)
        trec_corpus[parts[0]] = []

        trec_text_collection_data.append(parts[1])

        for w in parts[1].split(' '):
            ws = w.strip()
            if ws in vectors.vocab:
                trec_corpus[parts[0]].append(vectors.vocab[ws].index)

trec_text_collection = TextCollection(trec_text_collection_data)

print('all ', count, ' docs loaded')

# load topics file
trec_topics = {} # topic -> list of query term vector ids
max_topic_word_count = 0
with open(arg_topics_file, 'r') as inputFile:
    for line in inputFile:
        parts = line.split(' ', 1)

        if parts[0] not in trec_topics:
            trec_topics[parts[0]] = []

        for w in parts[1].split(' '):
            ws = w.strip()
            if ws in vectors.vocab:
                trec_topics[parts[0]].append(vectors.vocab[ws].index)

        if len(trec_topics[parts[0]]) > max_topic_word_count:
            max_topic_word_count = len(trec_topics[parts[0]])

print('all ', len(trec_topics), ' topics loaded')


print('creating histograms')
count = 0
# create histograms for every query term <-> doc term
# based on pairs from pre-ranked file, using the similarities of the wordembedding

# histogram file format: topicId DocId prerankscore numberOfTopicWords(N) idf1 idf2 ... idfN <hist1> <hist2> ... <histN>
with open('../data/'+arg_qrel_or_preranked+'_histogram_'+str(arg_bin_size)+'.txt', 'w') as outputFile:

    for topic, doc, score in topic_doc_pairs:
            count += 1
            if count % 10000 == 0:
                print('    ', count, ' ranked docs processed')

            if doc not in trec_corpus:
                print('skipping doc (not in corpus): '+ doc)
                continue
            if topic not in trec_topics:
                print('skipping topic (not in corpus): ' + doc)
                continue

            # get the word embedding ids
            topic_word_ids = trec_topics[topic]
            doc_words_ids = trec_corpus[doc]

            topic_vectors = np.array([vectors.word_vec(vectors.index2word[topic_word_ids[i]], True) for i in range(0,len(topic_word_ids))],np.float32)
            doc_vectors = np.array([vectors.word_vec(vectors.index2word[doc_words_ids[i]], True) for i in range(0,len(doc_words_ids))],np.float32)

            outputFile.write(topic+" "+doc+" "+str(score)+" "+str(len(topic_word_ids))+" ")
            for w in topic_word_ids:
                outputFile.write(str(trec_text_collection.idf(vectors.index2word[w])) + " ")

            qnum = len(topic_word_ids)
            d1_embed = topic_vectors
            d2_embed = doc_vectors

            curr_hist = cal_hist(d1_embed, d2_embed, qnum, arg_bin_size)
            curr_hist = curr_hist.tolist()
            outputFile.write(' '.join(map(str, curr_hist)))

            outputFile.write('\n')
            outputFile.flush()

print('Completed after (seconds): ', timeit.default_timer() - start_time)
print('Max topic words: ', max_topic_word_count)