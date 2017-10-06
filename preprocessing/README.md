## Preprocessing Pipeline

The following data needs to be available:

- TREC-8 documents
- TREC-Robust04 topics and qrels (ground truth) files
- Pre-ranked list of 2.000 results per topic (can be obtained with Lucene et al.)
- Word embedding (for example wikipedia with word2vec) in gensim model format

The following scripts have to be run, to create all needed input files for the neural network model.

- **transform_raw_trec_to_single_file.py** - input = trec directory, outputs to the ../data folder, removes html elements and non text characters, every doc = 1 line, starts with docId and then followed by whitespace separated text (stem option available)

- **transform_raw_topic_to_title_only.py** - input = topic file, output = in ../data (only takes the first title line), every topic = 1 line, topicId followed by whitespace separated text (stem option available)

- **generate_pair_histograms.py** - transformed doc data, transformed topic data, embedding file, qrel or prerank (depending on the train or test prep, has to be run for both once) outputs idf scores for topic terms + histogram information for every topic term -> doc, see the original paper for further information

- **create_train_test_folds.py** - splits the available topic -> doc pairs into 5-folds for cross validation and creates topic, relevant document, non relevant document pairs to orchestrate the input of the neural model.