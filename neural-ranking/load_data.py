import os
import numpy as np


def get_histsize_from_file(filepath):
    return int(os.path.basename(filepath).split('_')[0])


def load_data(filepath):
    histogramsize = get_histsize_from_file(filepath)
    data_per_topic = {} # topic -> (doc,score,[idf],[np.array(histogram)])

    count = 0
    with open(filepath,'r') as inputFile:
        for line in inputFile:

            count+=1
            if count == 10:
                break

            parts = line.strip().split()
            # histogram file format: topicId DocId prerankscore numberOfTopicWords(N) idf1 idf2 .. idfN <hist1> <hist2> ... <histN>
            topicId = parts[0]
            docId = parts[1]
            score = parts[2]

            numberOfTerms = int(parts[3])

            idfs = np.zeros((5,1), np.float32)
            histograms = []

            for i in range(numberOfTerms):
                idfs[i] = np.array([float(parts[i + 4])], np.float32)

            #idfs = np.array(idfs,np.float32)

            for i in range(numberOfTerms + 4, len(parts), histogramsize):
                hist = []
                for t in range(0, histogramsize):
                    hist.append(float(parts[i + t]))
                histograms.append(np.array(hist, np.float32))

            if topicId not in data_per_topic:
                data_per_topic[topicId] = []

            data_per_topic[topicId].append((docId,score,idfs,histograms))

    return data_per_topic
