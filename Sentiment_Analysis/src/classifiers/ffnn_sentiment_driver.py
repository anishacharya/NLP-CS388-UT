from Sentiment_Analysis.src.data_utils.definitions import SentimentExample
import Sentiment_Analysis.sentiment_config as conf

from common.utils.embedding import WordEmbedding, SentenceEmbedding
import common.common_config as common_conf
from common.utils.utils import pad_to_length

from typing import List
import numpy as np


def train_sentiment_ffnn(train_data: List[SentimentExample],
                         dev_data: List[SentimentExample],
                         word_embed: WordEmbedding) -> List[SentimentExample]:

    # Batch Data
    X = train_data

    raise NotImplementedError
