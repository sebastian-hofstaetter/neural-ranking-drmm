#
# Generate 5-fold split & 1-0 pairs for training and 5-fold split for testing
#
# Input:  qrels file (for training data), pre-ranked file (for test data) -> should be from the same test collection
# Output: (fixed) ../data/5-folds/<qrel-file-name>_fold_n.train
#                 ../data/5-folds/<qrel-file-name>_fold_n.test
#
# the split is created by the topic ids

import sys
import os
from random import randint

# make sure the argument is good (0 = the python file, 1 the actual argument)
if len(sys.argv) < 3:
    print('Needs 2 arguments - 1. the qrel file, 2. the prerank file')
    exit(0)

qrel_file = sys.argv[1]
qrel_name = os.path.basename(sys.argv[1]).replace('.txt','')
prerank_file = sys.argv[2]

#
# load qrels file
# ---------------------------------------------------------
#
qrels_doc_pairs = {} # topic -> (relevant: [docId], non-relevant: [docId]) (, doc id, relevant) relevant is boolean

count_rel = 0
count_non_rel = 0

with open(qrel_file, 'r') as inputFile:
    for line in inputFile:
        parts = line.split()

        if parts[0] not in qrels_doc_pairs:
            qrels_doc_pairs[parts[0]] = ([], [])

        if float(parts[3].strip()) > 0:
            qrels_doc_pairs[parts[0]][0].append(parts[2].strip())
            count_rel+=1
        else:
            qrels_doc_pairs[parts[0]][1].append(parts[2].strip())
            count_non_rel+=1

print(len(qrels_doc_pairs), ' topics loaded')
print(count_rel,' relevant docs loaded')
print(count_non_rel,' non-relevant docs loaded')

topic_ids = [id for id in sorted(qrels_doc_pairs)]

# here we could shuffle things around

part_size = int(round(len(topic_ids)/5,0))
part_1 = topic_ids[:part_size]
part_2 = topic_ids[part_size:2*part_size]
part_3 = topic_ids[2*part_size:3*part_size]
part_4 = topic_ids[3*part_size:4*part_size]
part_5 = topic_ids[4*part_size:]

print('part sizes 1-5:',len(part_1),len(part_2),len(part_3),len(part_4),len(part_5))
print('parts:\n[0] ',' '.join(part_1),'\n[1] ',' '.join(part_2),'\n[2] ',' '.join(part_3),'\n[3] ',' '.join(part_4),'\n[4] ',' '.join(part_5))

#
# load pre-ranked
# ---------------------------------------------------------
#

#
# load pre-ranked file
#
prerank_doc_pairs = {} # topic -> [docId]
count_prerank = 0
with open(prerank_file, 'r') as inputFile:
    for line in inputFile:
        parts = line.split()

        if parts[0] not in prerank_doc_pairs:
            prerank_doc_pairs[parts[0]] = []

        prerank_doc_pairs[parts[0]].append(parts[2].strip())
        count_prerank+=1

print(count_prerank, ' pre-ranked docs loaded')

#
# output helper functions
# ---------------------------------------------------------

# generate output per topic
# combine pairs for every rel - every non_rel
def create_1_0_pairs(topics):
    lines = []

    for topic in topics:
        for positive in qrels_doc_pairs[topic][0]:
            i = randint(0,len(qrels_doc_pairs[topic][1])-1)
            lines.append(topic + ' ' + positive + ' ' + qrels_doc_pairs[topic][1][i] + '\n')
            #i=1
            #for negative in qrels_doc_pairs[topic][1]:
            #    lines.append(topic+' '+positive+' '+negative+'\n')
            #    i+=1
            #    if i > 25:
            #        break

    print('\t got  ',len(lines),'train pairs')

    return lines

# combine the parts
def writeOutFiles(train1,train2,train3,train4,test,foldnumber):

    print('saving fold '+str(foldnumber))

    with open('../data/5-folds/'+str(qrel_name)+'_fold_'+str(foldnumber)+'.train', 'w') as trainFile:
        trainFile.writelines(create_1_0_pairs(train1))
        trainFile.writelines(create_1_0_pairs(train2))
        trainFile.writelines(create_1_0_pairs(train3))
        trainFile.writelines(create_1_0_pairs(train4))

    with open('../data/5-folds/' + str(qrel_name) + '_fold_' + str(foldnumber) + '.test', 'w') as testFile:
        lines = []
        for topic in test:
            for entry in prerank_doc_pairs[topic]:
                lines.append(topic + ' ' + entry+'\n')

        print('\t got  ', len(lines), 'test docs')
        testFile.writelines(lines)

#
# create the folds
# ---------------------------------------------------------

if not os.path.exists('../data/5-folds/'):
    os.makedirs('../data/5-folds/')

# train 1,2,3,4, test 5
writeOutFiles(part_1,part_2,part_3,part_4,part_5, 1)

# train 1,2,3,5, test 4
writeOutFiles(part_1,part_2,part_3,part_5,part_4, 2)

# train 1,2,4,5, test 3
writeOutFiles(part_1,part_2,part_4,part_5,part_3, 3)

# train 1,3,4,5, test 2
writeOutFiles(part_1,part_3,part_4,part_5,part_2, 4)

# train 2,3,4,5, test 1
writeOutFiles(part_2,part_3,part_4,part_5,part_1, 5)