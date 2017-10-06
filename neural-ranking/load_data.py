import os
import numpy as np


# helper to get histogram size from filename test_histogram_30.txt -> 30
def get_histsize_from_file(filepath):
    return int(os.path.basename(filepath).replace('.txt', '').split('_')[-1])


#
# loads the qrels rel-nonrel train fold file, + histogram data, returns keras ready numpy input, + empty labels
#
def get_keras_train_input(pair_file, histogram_file):
    topic_rel_nonrel = []
    with open(pair_file, 'r') as inputFile:
        for line in inputFile:

            parts = line.strip().split()
            topic_rel_nonrel.append((parts[0], parts[1], parts[2]))

            #if len(topic_rel_nonrel) == 100000:
            #    break

    print('loaded ' + str(len(topic_rel_nonrel)) + ' qrel pair entries')

    histogram_data,histogram_count = load_histogram_data(histogram_file)

    #
    # create numpy arrays
    #

    # the loss function needs a round number, * 2 because we have two input lines for every pair
    data_count = int(len(topic_rel_nonrel) / 10) * 10 * 2

    # np histogram
    histogram_input = np.zeros((data_count, 5, 30), dtype=np.float32)

    # topic idf
    idf_input = np.zeros((data_count, 5, 1), dtype=np.float32)

    # empty label array
    labels = np.zeros((data_count,), dtype=np.int32)
    labels[::2] = 1

    i_input = 0
    skipped_count = 0
    #
    # for every line here create 2 numpy lines, first line is relevant doc, second line is non_relevant
    #
    for i_output in range(0, data_count, 2):

        topic, rel_doc, nonrel_doc = topic_rel_nonrel[i_input]
        i_input += 1

        # there might be one or two pairs not in the histogram data - ignore them for now
        if topic in histogram_data and rel_doc in histogram_data[topic] and nonrel_doc in histogram_data[topic]:

            topic_rel_data = histogram_data[topic][rel_doc]
            topic_nonrel_data = histogram_data[topic][nonrel_doc]

            # histogram
            for w in range(len(topic_rel_data[2])):  # same topic -> therefore same histogram count
                histogram_input[i_output][w] = topic_rel_data[2][w]  # np.ones(30,dtype=np.float32)
                histogram_input[i_output + 1][w] = topic_nonrel_data[2][w]  # np.zeros(30,dtype=np.float32)

            # idf
            idf_input[i_output] = topic_rel_data[1]  # np.ones((5,1),dtype=np.float32) #
            idf_input[i_output + 1] = topic_nonrel_data[1]  # np.zeros((5,1),dtype=np.float32) #
        else:
            skipped_count += 1

    print("idf_input:",idf_input.shape)
    print("histogram_input:",histogram_input.shape)
    print("skipped_count:",skipped_count)

    return {'query': idf_input, 'doc': histogram_input}, labels


#
# loads the pre-ranked test fold file, + histogram data, returns keras ready numpy input + prerank data
#
def get_keras_test_input(preranked_file, histogram_file):
    topic_prerank = []
    with open(preranked_file, 'r') as inputFile:
        for line in inputFile:
            parts = line.strip().split()

            topic_prerank.append((parts[0], parts[1]))

    print('loaded ' + str(len(topic_prerank)) + ' prerank entries')

    histogram_data, histogram_count = load_histogram_data(histogram_file)

    #
    # create numpy arrays
    #

    data_count = len(topic_prerank)

    # np histogram
    histogram_input = np.zeros((data_count, 5, 30), dtype=np.float32)

    # topic idf
    idf_input = np.zeros((data_count, 5, 1), dtype=np.float32)

    i_input = 0
    skipped_count=0

    #
    # for every line here create 2 numpy lines, first line is relevant doc, second line is non_relevant
    #
    for i_output in range(0, data_count, 1):

        topic, rel_doc = topic_prerank[i_input]
        i_input += 1

        # there might be one or two pairs not in the histogram data - ignore them for now
        if topic in histogram_data and rel_doc in histogram_data[topic]:

            topic_rel_data = histogram_data[topic][rel_doc]

            # histogram
            for w in range(len(topic_rel_data[2])):  # same topic -> therefore same histogram count
                histogram_input[i_output][w] = topic_rel_data[2][w]  # np.ones(30,dtype=np.float32)

            # idf
            idf_input[i_output] = topic_rel_data[1]  # np.ones((5,1),dtype=np.float32) #
        else:
            skipped_count += 1

    print("idf_input:",idf_input.shape)
    print("histogram_input:",histogram_input.shape)
    print("skipped_count:",skipped_count)

    return {'query': idf_input, 'doc': histogram_input}, topic_prerank


def load_histogram_data(filepath):
    histogramsize = get_histsize_from_file(filepath)
    data_per_topic = {}  # topic -> doc -> (score,[idf],[np.array(histogram)])

    count = 0
    with open(filepath, 'r') as inputFile:
        for line in inputFile:

            #if count == 2000:
            #  break
            count += 1

            parts = line.strip().split()
            # histogram file format: topicId DocId prerankscore numberOfTopicWords(N) idf1 idf2 .. idfN <hist1> <hist2> ... <histN>
            topicId = parts[0]
            docId = parts[1]
            score = float(parts[2])

            numberOfTerms = int(parts[3])

            #
            # handle idfs
            #
            idfs = np.zeros((5, 1), np.float32)
            for i in range(numberOfTerms):
                idfs[i] = np.array([float(parts[i + 4])], np.float32)

            #
            # handle histogram data
            #
            histograms = []
            for i in range(numberOfTerms + 4, len(parts), histogramsize):
                hist = []
                for t in range(0, histogramsize):
                    hist.append(float(parts[i + t]))
                    # if t < 12 and float(parts[i + t]) > 0:
                    #    print('found hist',float(parts[i + t]),' at ',t,' for topic doc ',topicId, docId)
                histograms.append(np.array(hist, np.float32))

            if topicId not in data_per_topic:
                data_per_topic[topicId] = {}

            data_per_topic[topicId][docId] = (score, idfs, histograms)

    print('loaded ' + str(count) + ' topic<->doc histogram entries')
    return data_per_topic, count
