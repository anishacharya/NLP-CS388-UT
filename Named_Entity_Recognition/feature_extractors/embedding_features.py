def average_word_embedding(sentence, ix2embedding, word2ix):
    embed_vector = []
    for word in sentence:
        ix = word2ix[word]
        if ix is None:
            ix = word2ix['__UNK__']
        embed_vector = embed_vector + ix2embedding[ix]
    sentence_embedding = embed_vector / len(sentence)
    return sentence_embedding


def skip_thought(self):
    pass


def SIF(self):
    # https://github.com/PrincetonML/SIF_mini_demo
    pass
