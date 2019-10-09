from Sentiment_Classification.src.data_utils.definitions import SentimentExample
import Sentiment_Classification.sentiment_config as sentiment_config

from common.utils.embedding import WordEmbedding, SentenceEmbedding
from common.utils.utils import get_onehot_np, pad_to_length
import common.common_config as common_conf

from typing import List
import numpy as np
import torch


def get_xy_embedded(data: List[SentimentExample], word_embed: WordEmbedding):
    """ From input batch extracts and return sentences and labels batch"""
    # we will convert each sentence into its sentence embedding using DAN
    # we will also get labels
    sentence_embed = SentenceEmbedding(word_embed=word_embed)
    x = np.zeros((len(data), word_embed.emb_dim), dtype=np.float32)
    y = np.zeros((1, len(data)), dtype=np.int32)    # np.eye won't work for float which we use to get one hot

    for ix, data_point in enumerate(data):
        sentence = data_point.indexed_words
        sentence_embedding = sentence_embed.average_word_embedding(sentence=sentence,
                                                                   word_dropout_rate=sentiment_config.word_dropout_rate)
        x[ix, :] = sentence_embedding
        y[:, ix] = data_point.label

    x = torch.from_numpy(x)

    y_onehot_np = get_onehot_np(y=y, no_classes=sentiment_config.no_classes)
    y_onehot = torch.from_numpy(y_onehot_np)
    return x, y_onehot


def get_xy_padded(data: List[SentimentExample], word_embed: WordEmbedding):
    seq_len = sentiment_config.seq_max_len
    pad_ix = word_embed.word_ix.add_and_get_index(common_conf.PAD_TOKEN)

    x = np.ones((len(data), seq_len), dtype=np.float32)
    y = np.zeros((1, len(data)), dtype=np.int32)  # np.eye won't work for float which we use to get one hot

    for ix, data_point in enumerate(data):
        sentence = data_point.indexed_words
        x[ix, :] = pad_to_length(np_arr=sentence, pad_ix=pad_ix)
        y[:, ix] = data_point.label
    x = torch.from_numpy(x)

    y_onehot_np = get_onehot_np(y=y, no_classes=sentiment_config.no_classes)
    y_onehot = torch.from_numpy(y_onehot_np)
    return x, y_onehot
