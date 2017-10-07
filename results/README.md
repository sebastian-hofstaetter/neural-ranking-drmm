# Experiments

### #Schirra

**Commit:** 84d899d4aa23e432f4b4d08b1342d12337a9b4db
 
**Goal**: Fix errors
 
- Ground truth now based on qrels file not preranked
- Results of the neural network look like ~ 1 = relevant, ~ -1, non-relevant, but the network does not learn "scores"
    - In the paper it only says "re-rank" => multiplied the pre-ranked scores with neural network results (should filter out non-relevant docs(?))
     
**Training**: Trained 5-fold cross-validation for 30 epochs (training loss flattened at 0.5)

**Result**: MAP Robust04: **0.2600** (The goal was to reach 0.279) At least we reached the baseline now.

**Notes**:
 - Maybe longer training improves the results, but it looks like the loss is not decreasing.
 
### #Shepard

**Commit:** 514952a00a964e7bbe72f3879e922ba22e27cccb
 
**Goal**: Replicate the implementation & results presented in the original paper
 
- Histogram (30 bins) smoothed with log (used histogram formula from Matchzoo), but used wikipedia stemmed word vectors
- OOV words ignored for now, because only 3 OOV words are in the Robust04 topic titles
- Porter stemmer was used instead of Krovetz stemmer, Nltk stopword list instead of INQUERY
- The document (high, low) pairing was not clear from the paper ->  a 50 top / 50 bottom split of ranked documents was conducted and and pairs built from same indices in the two lists

**Training**: Trained 5-fold cross-validation for 30 epochs (training loss flattened at 0.73 - 0.74)

**Result**: MAP Robust04: **0.2146** (The goal was to reach 0.279)

**Notes**:
 - The first 12 histogram values are always empty (is there a problem with the formula, min-max of the similarity?)