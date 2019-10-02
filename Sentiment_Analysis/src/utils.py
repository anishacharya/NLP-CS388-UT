from Sentiment_Analysis.src.data_utils.definitions import SentimentExample
import Sentiment_Analysis.sentiment_config as sentiment_config
from common.utils.embedding import WordEmbedding, SentenceEmbedding
from common.utils.utils import get_onehot_np

from typing import List
import numpy as np
import torch


def get_xy(data: List[SentimentExample], word_embed: WordEmbedding):
    """ From input batch extracts and return sentences and labels batch"""
    # we will convert each sentence into its sentence embedding using DAN
    # we will also get labels
    sentence_embed = SentenceEmbedding(word_embed=word_embed)
    x = np.zeros((len(data), word_embed.emb_dim), dtype=np.float32)
    y = np.zeros((1, len(data)), dtype=np.int64)

    for ix, data_point in enumerate(data):
        sentence = data_point.indexed_words
        sentence_embedding = sentence_embed.average_word_embedding(sentence=sentence)
        x[ix, :] = sentence_embedding
        y[:, ix] = data_point.label

    x = torch.from_numpy(x).float()
    y_onehot = torch.from_numpy(get_onehot_np(y=y, no_classes=sentiment_config.no_classes))

    return x, y_onehot
