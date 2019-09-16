from typing import List


def word_embedding(word, ix2embed, word2ix):
    ix = word2ix[word]
    if ix in ix2embed:
        word_embed = ix2embed[ix]
    else:
        word_embed = ix2embed[word2ix['__UNK__']]
    return word_embed


def average_word_embedding(sentence: List, ix2embedding, word2ix):
    embed_vector = [0] * 300
    for word in sentence:
        if word in word2ix:
            ix = word2ix[word]
        else:
            ix = word2ix['__UNK__']
        if ix in ix2embedding:
            embed_vector = [sum(x) for x in zip(embed_vector, ix2embedding[ix])]
        else:
            embed_vector = [sum(x) for x in zip(embed_vector, ix2embedding[word2ix['__UNK__']])]

    sentence_embedding = [i/len(sentence) for i in embed_vector]
    return sentence_embedding


def get_context_vector(tokens: List, idx, window_len, word2ix, ix2embed, left=False, right=False):
    """
    Creates a padded seq and gives windowed sentence embedding using average
    """
    padded_seq = []
    # pad before
    for i in range(0, window_len):
        padded_seq.append('__UNK__')
    # append tokens
    for token in tokens:
        padded_seq.append(token.lower() if token is not '__UNK__' else token)
    # pad after
    for i in range(0, window_len):
        padded_seq.append('__UNK__')

    ix_curr = idx + window_len
    ix_lb = ix_curr - window_len
    ix_ub = ix_curr + window_len

    if left is True:
        sentence = padded_seq[ix_lb:ix_curr + 1]
        return average_word_embedding(sentence=sentence, ix2embedding=ix2embed, word2ix=word2ix)

    if right is True:
        sentence = padded_seq[ix_curr:ix_ub + 1]
        return average_word_embedding(sentence=sentence, ix2embedding=ix2embed, word2ix=word2ix)

    sentence = padded_seq[ix_lb:ix_ub + 1]
    return average_word_embedding(sentence=sentence, ix2embedding=ix2embed, word2ix=word2ix)


def skip_thought(self):
    pass


def sif_arora(self):
    # https://github.com/PrincetonML/SIF_mini_demo
    pass
