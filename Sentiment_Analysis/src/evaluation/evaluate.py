from common.evaluation.evaluate_classifier import ClassificationEval
from common.utils.embedding import WordEmbedding
from Sentiment_Analysis.src.data_utils.definitions import SentimentExample
from Sentiment_Analysis.src.utils import get_xy_FFNN
from common.utils.utils import argmax_from_onehot
import numpy as np


def evaluate_sentiment(model, data: [SentimentExample], word_embedding:WordEmbedding, model_type: str):
    if model_type == 'FFNN':
        x, y = get_xy_FFNN(data=data, word_embed=word_embedding)
    else:
        raise NotImplementedError
    pred_prob = model(x)
    y_ground = np.zeros(len(data))
    y_pred = np.zeros(len(data))
    for ix, ex in enumerate(data):
        y_ground[ix] = int(ex.label)
    for ix, prob in enumerate(pred_prob):
        y_pred[ix] = argmax_from_onehot(prob)

    metrics = ClassificationEval(ground_truth=y_ground, prediction=y_pred)

    return y_pred, metrics
