## Hello,

this repository contains an implementation of the DRMM model.

**Original paper**: A Deep Relevance Matching Model for Ad-hoc Retrieval (Guo et al.) [https://dl.acm.org/citation.cfm?id=2983769](https://dl.acm.org/citation.cfm?id=2983769)

Some code is taken from [MatchZoo](https://github.com/faneshion/MatchZoo) (referenced in the respective files).

## Table of Contents

- **Preprocessing**: Contains all preprocessing code (starting from raw TREC data) to create all input files for the neural ranking model (this includes generating the histogram, idf informations). See the [peprocessing Readme](preprocessing/README.md) for more.

- **Neural Ranking**: Contains the neural ranking model

- **Results**: Contains a description of experiments & their raw and trec_eval evaluated results.

## Dependencies

- TREC 8 corpus data (not preprocessed: fbis,fr94,ft,latimes)
- Python 3: Latest Keras and Tensorflow