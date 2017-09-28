from math import log


class TextCollection(object):

    """
    FROM: http://www.nltk.org/_modules/nltk/text.html#TextCollection

    Just used to create the idf score
    """
    def __init__(self, source):

        self._texts = source
        self._idf_cache = {}

    def tf(self, term, text):
        """ The frequency of the term in text. """
        return text.count(term) / len(text)


    def idf(self, term):
        """ The number of texts in the corpus divided by the
        number of texts that the term appears in.
        If a term does not appear in the corpus, 0.0 is returned. """
        # idf values are cached for performance.
        idf = self._idf_cache.get(term)
        if idf is None:
            matches = len([True for text in self._texts if term in text])
            # FIXME Should this raise some kind of error instead?
            idf = (log(len(self._texts) / matches) if matches else 0.0)
            self._idf_cache[term] = idf
        return idf


    def tf_idf(self, term, text):
        return self.tf(term, text) * self.idf(term)