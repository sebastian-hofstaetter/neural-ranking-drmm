#!/bin/bash

foldname="grissom_fold_"

trainFoldLocation="../data/5-folds/robust04.qrels_fold_"
trainHistogramLocation="../data/qrel_histogram_30.txt"
testFoldLocation="../data/5-folds/robust04.qrels_fold_"
testHistogramLocation="../data/prerank_histogram_30.txt"


for number in {1..5}
do
	#echo "running fold: $number"
	python3 run_model.py "$foldname$number" "$trainFoldLocation$number.train" "$trainHistogramLocation" "$testFoldLocation$number.test" "$testHistogramLocation" #&> logs/fold1.log &
done
exit 0

#python3 run_model.py schirra_fold1 "../data/5-folds/robust04.qrels_fold_1.train" "../data/qrel_histogram_30.txt" "../data/5-folds/robust04.qrels_fold_1.test" "../data/prerank_histogram_30.txt" #&> logs/fold1.log &
#
#python3 run_model.py schirra_fold2 "../data/5-folds/robust04.qrels_fold_2.train" "../data/qrel_histogram_30.txt" "../data/5-folds/robust04.qrels_fold_2.test" "../data/prerank_histogram_30.txt" #&> logs/fold2.log &
#python3 run_model.py schirra_fold3 "../data/5-folds/robust04.qrels_fold_3.train" "../data/qrel_histogram_30.txt" "../data/5-folds/robust04.qrels_fold_3.test" "../data/prerank_histogram_30.txt" #&> logs/fold3.log &
#python3 run_model.py schirra_fold4 "../data/5-folds/robust04.qrels_fold_4.train" "../data/qrel_histogram_30.txt" "../data/5-folds/robust04.qrels_fold_4.test" "../data/prerank_histogram_30.txt" #&> logs/fold4.log &
#python3 run_model.py schirra_fold5 "../data/5-folds/robust04.qrels_fold_5.train" "../data/qrel_histogram_30.txt" "../data/5-folds/robust04.qrels_fold_5.test" "../data/prerank_histogram_30.txt" #&> logs/fold5.log &
#
#
#wait

#echo "all processes completed"