#
# Generate 5-fold split for training and testing from histogram file
#
# Input:  ../data/trec_corpus__histogram_<bin count>.txt
# Output: (fixed) ../data/n-folds/<bin count>_histogram_fold_n.train
#                 ../data/n-folds/<bin count>_histogram_fold_n.test

import sys
import os

# make sure the argument is good (0 = the python file, 1 the actual argument)
if len(sys.argv) < 2:
    print 'Needs 1 argument - the histogram file'
    exit(0)

histogram_file = sys.argv[1]
histogram_bins = int(os.path.basename(histogram_file).split('_')[-1].replace('.txt',''))

with open(histogram_file, 'r') as inputFile:
    data = inputFile.readlines()

part_size = len(data)/5
part_1 = data[:part_size]
part_2 = data[part_size:2*part_size]
part_3 = data[2*part_size:3*part_size]
part_4 = data[3*part_size:4*part_size]
part_5 = data[4*part_size:]

print 'part sizes 1-5:',len(part_1),len(part_2),len(part_3),len(part_4),len(part_5)

if not os.path.exists('../data/5-folds/'):
    os.makedirs('../data/5-folds/')


# combine the parts
def writeOutFiles(train1,train2,train3,train4,test,foldnumber):

    print 'saving fold '+str(foldnumber)

    with open('../data/5-folds/'+str(histogram_bins)+'_histogram_fold_'+str(foldnumber)+'.train', 'w') as trainFile:
        trainFile.writelines(train1)
        trainFile.writelines(train2)
        trainFile.writelines(train3)
        trainFile.writelines(train4)

    with open('../data/5-folds/' + str(histogram_bins) + '_histogram_fold_' + str(foldnumber) + '.test', 'w') as testFile:
        testFile.writelines(test)


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