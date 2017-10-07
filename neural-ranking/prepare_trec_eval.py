import sys

filepath = sys.argv[1]
filepath_out = sys.argv[2]

preranked_file = sys.argv[3]

preranked = [] # (topic, doc id, score) score is 0,1 if qrel

with open(preranked_file, 'r') as inputFile:
    for line in inputFile:
        parts = line.split()
        preranked.append((int(parts[0]), parts[2].strip(), parts[4].strip()))


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

# prerank * neural score

i_rank =0
for topic in sorted(data):
    list = data[topic]
    for i in range(len(list)):
        score, docId = list[i]

        preranked_item = preranked[i_rank]

        if preranked_item[0] != topic:
            print('fixing topic unequal',preranked_item[0],topic)
            if preranked_item[0] < topic:
                while preranked[i_rank][0] < topic:
                    i_rank+=1
            preranked_item = preranked[i_rank]

        if preranked_item[0] != topic or preranked_item[1] != docId:
            print('aaahh!!',topic,preranked_item[0],docId,preranked_item[1])
            preranked_score = next(item[2] for item in preranked if item[0] == topic and item[1] == docId)

        else:
            preranked_score = preranked_item[2] #next(item[2] for item in preranked if item[0] == topic and item[1] == docId)

        #preranked_score = filter(lambda item: item[0] == topic and item[1] == docId, preranked)

        list[i] = (score*float(preranked_score),docId)
        i_rank+=1



with open(filepath_out ,'w') as outFile:
    for topic in sorted(data):
        i = 0
        for tuple in sorted(data[topic],reverse=True):
            if i == 1000:
                break
            outFile.write(str(topic)+'\t0\t'+tuple[1]+'\t'+str(i)+'\t'+str(tuple[0])+'\tdrmm\n')
            i+=1
