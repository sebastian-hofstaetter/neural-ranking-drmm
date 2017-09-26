#
# Trec topic file title extractor
#
# Input:  Trec topics file (title line = 1 line), optional: "doStem" as 2nd argument if words should be stemmed
# Output: single file in ../data/<name of the input file>
#         contents: num1 text text text
#                   num2 text text
#         (text contains no newlines, num is the topic number)
#
# The text is stripped of all non text characters, lower-cased, and only single whitespaces
#

import os
import sys
import re
from nltk.stem import *

# make sure the argument is good (0 = the python file, 1 the actual argument)
if len(sys.argv) < 2 or not os.path.isfile(sys.argv[1]):
    print 'Needs 1 argument - the trec topic file path!'
    exit(0)

cleanTextRegex = re.compile('[^a-zA-Z]')
count = 0
stemmer = PorterStemmer()
doStem = len(sys.argv) == 3 and sys.argv[2] == 'doStem'

outFilepath = '../data/'+os.path.basename(sys.argv[1])
if doStem:
    outFilepath = '../data/' + os.path.splitext(os.path.basename(sys.argv[1]))[0] +".stemmed"\
                  +os.path.splitext(os.path.basename(sys.argv[1]))[1]

with open(outFilepath, 'w') as outputFile:
    with open(sys.argv[1], 'r') as inputFile:
        currentId = ''
        for inLine in inputFile.readlines():
            if inLine.startswith('<num> Number:'):
                currentId = inLine.replace('<num> Number:', '').strip()
            if inLine.startswith('<title>'):
                text = inLine.replace('<title>', '').strip()
                # clean text
                text = cleanTextRegex.sub(' ', text).lower()
                # remove multiple whitespaces
                text = text.replace('    ',' ').replace('   ',' ').replace('  ',' ')

                wordList = []
                for w in text.split(' '):
                    if w:
                        if doStem:
                            cleaned = stemmer.stem(w.strip())
                        else:
                            cleaned = w.strip()
                        wordList.append(cleaned)
                outputText = ' '.join(wordList)

                # write single line output
                outputFile.write(currentId)
                outputFile.write(' ')
                outputFile.write(outputText.strip())
                outputFile.write('\n')
                count = count + 1

print 'Completed all ', count, ' topics'
print 'Saved in: ', outFilepath
