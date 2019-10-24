from common.evaluation.evaluate_classifier import ClassificationEval, binary_accuracy
from common.utils.embedding import WordEmbedding
from sentiment_classification.src.data_utils.definitions import SentimentExample
from sentiment_classification.src.utils import get_xy_embedded, get_xy_padded
from common.utils.utils import argmax_from_onehot
import numpy as np


def evaluate_sentiment_simple(model, data: [SentimentExample], word_embedding: WordEmbedding, model_type: str):
    if model_type == 'FFNN':
        x, y = get_xy_embedded(data=data, word_embed=word_embedding)
    elif model_type == 'RNN':
        x, y = get_xy_padded(data=data, word_embed=word_embedding)
    else:
        raise NotImplementedError

    y_hat = model(x)
    y_pred = np.zeros(len(data))
    for ix, prob in enumerate(y_hat):
        y_pred[ix] = argmax_from_onehot(prob)

    acc = binary_accuracy(y_hat=y_hat, y=y)
    return y_pred, acc


def evaluate_sentiment(model, data: [SentimentExample], word_embedding: WordEmbedding, model_type: str):
    if model_type == 'FFNN':
        x, y = get_xy_embedded(data=data, word_embed=word_embedding)
    elif model_type == 'RNN':
        x, y = get_xy_padded(data=data, word_embed=word_embedding)
    else:
        raise NotImplementedError
    pred_prob = model(x)

    y_ground = np.zeros(len(data))
    y_pred = np.zeros(len(data))
    for ix, ex in enumerate(data):
        y_ground[ix] = ex.label
    for ix, prob in enumerate(pred_prob):
        y_pred[ix] = argmax_from_onehot(prob)

    metrics = ClassificationEval(ground_truth=y_ground.astype(int), prediction=y_pred.astype(int))

    return y_pred, metrics
