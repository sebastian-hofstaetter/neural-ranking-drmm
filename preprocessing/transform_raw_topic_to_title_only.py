#
# Trec topic file title extractor
#
# Input:  Trec topics file (title line = 1 line)
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

# make sure the argument is good (0 = the python file, 1 the actual argument)
if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
    print 'Needs 1 argument - the trec topic file path!'
    exit(0)

cleanTextRegex = re.compile('[^a-zA-Z]')
count = 0
outFilepath = '../data/'+os.path.basename(sys.argv[1])

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

                # write single line output
                outputFile.write(currentId)
                outputFile.write(' ')
                outputFile.write(text.strip())
                outputFile.write('\n')
                count = count + 1

print 'Completed all ', count, ' topics'
print 'Saved in: ', outFilepath
