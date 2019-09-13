from data_utils.nerdata import PersonExample
from feature_extractors.ner_features import NERFeatureExtractors, create_index, \
    sparse_feature_encoder, sparse_feature_decoder, Feature
from utils.utils import Indexer, sigmoid, logistic_regression_loss_imbalanced, flatten
from utils.optimizers import UnregularizedAdagradTrainer

import numpy as np
from typing import List
from random import shuffle
import time
from collections import Counter


class BaselineNERClassifier(object):
    """
    Classifier to classify a token in a sentence as a PERSON token or not.
    Constructor arguments are merely suggestions; you're free to change these.
    """

    def __init__(self, weights: np.ndarray, word_ix: Indexer, pos_ix: Indexer):
        self.w = weights
        self.word_ix = word_ix
        self.pos_ix = pos_ix

    def predict(self, tokens: List[str], idx: int):
        """
        Makes a prediction for token at position idx in the given PersonExample
        :param tokens:
        :param idx:
        :return: 0 if not a person token, 1 if a person token
        """
        threshold = 0.6

        token = tokens[idx]
        if token.word not in self.word_ix.objs_to_ints:
            return 0

        projector = NERFeatureExtractors(word_index=self.word_ix,
                                         pos_index=self.pos_ix)
        feat_space_unigram = projector.uni_gram_index_feature_per_token(token=token)
        feat_space_pos = projector.pos_index_feature_per_token(token=token)
        feat_space = np.concatenate((feat_space_unigram, feat_space_pos), axis=None)

        prediction = sigmoid(np.matmul(feat_space, self.w))

        if prediction > threshold:
            return 1
        return 0


# def get_non_zero_gradients_ix(x):
#     non_zero_pos = []
#     dim = x.shape
#     for i in range(dim[0]):
#         ixs = list(np.nonzero(x[i, :])[0])
#         non_zero_pos.append(ixs)
#     return non_zero_pos
#
#
# def expand_non_zero_grad(sparse_x, feat_dim):
#     feat = np.zeros(len(sparse_x), feat_dim)
#     for row in range(len(sparse_x)):
#         for ix in sparse_x[row]:
#             feat[row, ix] = 1
#     return feat


def get_encoded_batch(sparse_feature_full, curr_batch, next_batch):
    sparse_feat_batch = []
    for feat in sparse_feature_full:
        feat_batch = feat.feature[curr_batch:next_batch]
        sparse_feat_batch.append(Feature(feature=feat_batch, size=feat.size))

    total_tokens_batch = len(flatten(sparse_feat_batch[0].feature))
    return sparse_feat_batch, total_tokens_batch


def train_model_based_ner(ner_exs: List[PersonExample]):
    shuffle(ner_exs)
    word_ix, pos_ix = create_index(ner_exs)
    ner_feature_extractor = NERFeatureExtractors(word_index=word_ix,
                                                 pos_index=pos_ix)

    W = np.random.rand(ner_feature_extractor.feature_dim)
    optim = UnregularizedAdagradTrainer(W)
    upweight_alpha = 1

    epochs = 10
    batch_size = 10
    print_iter = 500

    no_of_batches = len(ner_exs) // batch_size

    # First create and store the sparse features efficiently so that we can use them at each batch
    sparse_features_encoded, y_train = sparse_feature_encoder(ner_exs=ner_exs, word_ix=word_ix, pos_ix=pos_ix)

    for epoch in range(epochs):
        t = time.time()
        curr_batch = 0
        running_loss = 0

        for i in range(no_of_batches):
            next_batch = (curr_batch + batch_size)

            # get batch code
            sparse_features_encoded_batch, total_data_points = get_encoded_batch(
                sparse_feature_full=sparse_features_encoded,
                curr_batch=curr_batch,
                next_batch=next_batch)
            y_train_batch = y_train[curr_batch: curr_batch+total_data_points]

            # sparse_features_encoded_batch = sparse_features_encoded[curr_batch: next_batch]
            x_train_batch, sparse_grad_loc = sparse_feature_decoder(sparse_features_encoded_batch)

            # data_batch = ner_exs[curr_batch: next_batch]
            # x_train_batch, y_train_batch = project_to_continuous_space(ner_exs=data_batch,
            #                                                           word_ix=word_ix,
            #                                                           pos_ix=pos_ix)

            curr_batch = next_batch
            x1 = np.matmul(x_train_batch, W)
            y_pred = sigmoid(x1)

            # sparse_grad_loc = get_non_zero_gradients_ix(x_train_batch)
            grad = y_train_batch - y_pred

            grad_counter = Counter()

            for j in range(len(sparse_grad_loc)):
                beta = 1
                if y_train_batch[j] == 1:
                    beta = upweight_alpha
                for ix in sparse_grad_loc[j]:
                    grad_counter[ix] += grad[j] * x_train_batch[j, ix] * beta * 10

            loss = logistic_regression_loss_imbalanced(y_true=y_train_batch,
                                                       y_hat=y_pred,
                                                       alpha=upweight_alpha)

            optim.apply_gradient_update(gradient=grad_counter,
                                        batch_size=batch_size)
            running_loss += loss
            if i % print_iter == 0:
                print('===== Current Loss ====', running_loss/(i+1))
        print("==== Time Taken for this epoch===", time.time() - t)

    return BaselineNERClassifier(W, word_ix=word_ix, pos_ix=pos_ix)





