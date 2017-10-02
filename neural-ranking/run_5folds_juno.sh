python3 run_model.py fold1 "../data/5-folds/30_histogram_fold_1.train" "../data/5-folds/30_histogram_fold_1.test" &> logs/fold1.log &
python3 run_model.py fold2 "../data/5-folds/30_histogram_fold_2.train" "../data/5-folds/30_histogram_fold_2.test" &> logs/fold2.log &
python3 run_model.py fold3 "../data/5-folds/30_histogram_fold_3.train" "../data/5-folds/30_histogram_fold_3.test" &> logs/fold3.log &
python3 run_model.py fold4 "../data/5-folds/30_histogram_fold_4.train" "../data/5-folds/30_histogram_fold_4.test" &> logs/fold4.log &
python3 run_model.py fold5 "../data/5-folds/30_histogram_fold_5.train" "../data/5-folds/30_histogram_fold_5.test" &> logs/fold5.log &

wait

echo "all processes completed"