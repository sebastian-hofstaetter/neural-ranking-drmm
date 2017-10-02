import sys

filepath = sys.argv[1]
filepath_out = sys.argv[2]

data = {}

with open(filepath ,'r') as inputFile:
    for line in inputFile:
        parts = line.split()

        topicId = int(parts[0])
        docId = parts[1]
        score = float(parts[2])

        if topicId not in data:
            data[topicId] = []

        data[topicId].append((score, docId))

with open(filepath_out ,'w') as outFile:
    for topic in sorted(data):
        i = 0
        for tuple in sorted(data[topic],reverse=True):
            outFile.write(str(topic)+'\t0\t'+tuple[1]+'\t'+str(i)+'\t'+str(tuple[0])+'\tdrmm\n')
            i+=1
