#
# Trec data to single file transformer
#
# Input:  Directory of raw trec corpus data (subdirectory structure possible), optional: "doStem" as 2nd argument if words should be stemmed
# Output: single file in ../data/trec_corpus.txt
#         contents: id1 text text text
#                   id2 text text
#         (text contains no newlines, id can contain special chars except whitespaces)
#
# The text is stripped of all html tags, all non text characters, lower-cased, and only single whitespaces
#

import os
import timeit
import sys
import re
import codecs
from nltk.stem import *

# make sure the argument is good (0 = the python file, 1 the actual argument)
if len(sys.argv) < 2 or not os.path.isdir(sys.argv[1]):
    print 'Needs 1 argument - the trec data directory path!'
    exit(0)

cleanTextRegex = re.compile('[^a-zA-Z]')
cleanHtmlRegex = re.compile('<[^<]+?>')

docCount = 0
stemmer = PorterStemmer()
doStem = len(sys.argv) == 3 and sys.argv[2] == 'doStem'

def handleTrecFile(filename, outputFile):
    """
    handles a single trec file transform, writes to the output file
    """
    global docCount
    with codecs.open(filename, "r", "iso-8859-1") as f:
        contents = f.readlines()
        currentDocId = ''
        currentDocContent = []
        recordContent = False
        for line in contents:

            # ignore empty lines
            if line.isspace():
                continue

            # get the current document id
            if line.startswith('<DOCNO>'):
                currentDocId = line.replace('<DOCNO>', '').replace('</DOCNO>', '').strip()
                recordContent = True
                continue

            # process end of document
            if line.startswith('</DOC>'):

                # clean the html tags out
                parsed = cleanHtmlRegex.sub(' ', ' '.join(currentDocContent))
                # clean non text characters
                parsed = cleanTextRegex.sub(' ', parsed)
                # clean whitespaces + lower words + concat again
                wordList = []
                for w in parsed.split(' '):
                    if w:
                        if doStem:
                            cleaned = stemmer.stem(w.lower().strip())
                        else:
                            cleaned = w.lower().strip()
                        wordList.append(cleaned)
                outputText = ' '.join(wordList)

                # write single line output, reset state
                outputFile.write(currentDocId)
                outputFile.write(' ')
                outputFile.write(outputText)
                outputFile.write('\n')
                recordContent = False
                currentDocContent = []
                docCount = docCount + 1
                continue

            # we are inside a document - record !
            if recordContent:
                currentDocContent.append(line)

        outputFile.flush()


#
# gather trec files and transform them one by one
#
count = 0
start_time = timeit.default_timer()
outFileName = '../data/trec_corpus.txt'
if doStem:
    outFileName = '../data/trec_corpus.stemmed.txt'

with open(outFileName, 'w') as outputFile:
    for dirpath, dirnames, fileNames in os.walk(sys.argv[1]):
        for file in fileNames:
            handleTrecFile(dirpath + os.sep + file, outputFile)
            count = count + 1
            if count % 10 == 0:
                print 'Completed ', count, ' files, with ', docCount, ' docs, time:', timeit.default_timer() - start_time

print '\n-------\n', 'Completed all ', count, ' files, with ', docCount, ' docs, time: ', timeit.default_timer() - start_time
