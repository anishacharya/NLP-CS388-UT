class SentimentExample:
    """
    Wraps a sequence of word indices with a label
    ex: for RT: 0-1 label (0 = negative, 1 = positive)
    """
    def __init__(self, indexed_words, label: int):
        self.indexed_words = indexed_words
        self.label = label

    def __repr__(self):
        return repr(self.indexed_words) + "; label=" + repr(self.label)

    def get_indexed_words_reversed(self):
        return [self.indexed_words[len(self.indexed_words) - 1 - i] for i in range(0, len(self.indexed_words))]

