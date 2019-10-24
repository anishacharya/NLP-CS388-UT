from src.data_utils.definitions import Indexer, LabeledSentence, Token
from src.feature_extractors.emission_features import extract_emission_features
from src.models.emmission_crf import EmissionCrfNerModel
from src.evaluation.ner_eval import print_evaluation_metric, write_test_output
import src.config as conf
from src.data_utils.utils import get_word_index
from src.models.utils import prepare_label_point
import numpy as np
from collections import Counter
import torch
import time


# stops = set(stopwords.words("english"))
# stops = set(punctuation)
# stops.update(set(punctuation))
# stops.update({'-X-', ',', '$', ':', '-DOCSTART-'})
stops = set()


# Trains a CrfNerModel on the given corpus of sentences.
def train_emission_crf_ner(train_data: [LabeledSentence], dev_data: [Token], test_data):
    tag_indexer = Indexer()
    word_indexer = Indexer()
    word_counter = Counter()

    tag_indexer.add_and_get_index(conf.PAD_TOKEN)   # padding
    word_indexer.add_and_get_index(conf.PAD_TOKEN)
    tag_indexer.add_and_get_index(conf.EOS_TOKEN)   # End of Sentence
    word_indexer.add_and_get_index(conf.EOS_TOKEN)
    tag_indexer.add_and_get_index(conf.BOS_TOKEN)   # Beginning of Sentence
    word_indexer.add_and_get_index(conf.BOS_TOKEN)
    tag_indexer.add_and_get_index(conf.UNK_TOKEN)   # Unk Words
    word_indexer.add_and_get_index(conf.UNK_TOKEN)

    for sentence in train_data:
        for token in sentence.tokens:
            word_counter[token.word] += 1.0

    for sentence in train_data:
        for token in sentence.tokens:
            # If the word occurs fewer than two times, don't index it -- we'll treat it as UNK
            get_word_index(word_indexer=word_indexer, word_counter=word_counter, stops=stops, word=token.word, th=0)
        for tag in sentence.get_bio_tags():
            tag_indexer.add_and_get_index(tag)

    feature_indexer = Indexer()
    # 4-d list indexed by sentence index, word index, tag index, feature index
    feature_cache = [[[[] for k in range(0, len(tag_indexer))] for j in
                      range(0, len(train_data[i]))] for i in range(0, len(train_data))]
    for sentence_idx in range(0, len(train_data)):
        if sentence_idx % 100 == 0:
            print("Ex %i/%i" % (sentence_idx, len(train_data)))
        for word_idx in range(0, len(train_data[sentence_idx])):
            for tag_idx in range(0, len(tag_indexer)):
                feature_cache[sentence_idx][word_idx][tag_idx] = \
                    extract_emission_features(train_data[sentence_idx].tokens, word_idx,
                                              tag_indexer.get_object(tag_idx), feature_indexer, add_to_indexer=True)

    crf_model = EmissionCrfNerModel(word_ix=word_indexer, tag_ix=tag_indexer,
                                    feature_cache=feature_cache, feature_dim=len(feature_indexer),
                                    feature_ix=feature_indexer)

    lr = conf.initial_lr
    optimizer = torch.optim.Adam([crf_model.emission_weights, crf_model.transitions], lr=lr)
    best_f1 = 0
    for epoch in range(conf.epochs):
        ti = time.time()
        total_loss = 0
        for ix, sentence in enumerate(train_data):
            x = np.array(feature_cache[ix])
            y = prepare_label_point(sentence, tag_indexer).unsqueeze(0)
            crf_model.zero_grad()
            loss = crf_model.nll(x, y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            if ix % 100 == 0:
                print("Current Loss =", total_loss/(ix+1))
        # Compute Dev Acc.
        print("-------------------------")
        print("Epoch: ", epoch)
        print("Time taken: ", time.time()-ti)
        print("current Loss =", total_loss/len(train_data))
        dev_decoded = [crf_model.decode(test_ex.tokens) for test_ex in dev_data]
        f1 = print_evaluation_metric(dev_data, dev_decoded)
        if f1 > best_f1:
            test_decoded = [crf_model.decode(test_ex.tokens) for test_ex in test_data]
            write_test_output(test_decoded, conf.output_path)
        print(" ")
        print(" -------------------------")

        # lr = lr/2
        # if (epoch + 1) % 5 == 0:
        #    lr = conf.initial_lr
    return crf_model
