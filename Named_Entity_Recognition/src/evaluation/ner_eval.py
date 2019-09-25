from src.data_utils.definitions import LabeledSentence
from typing import List


def print_evaluation_metric(gold_sentences: List[LabeledSentence], guess_sentences: List[LabeledSentence]):
    """
    Evaluates the guess sentences with respect to the gold sentences
    :param gold_sentences:
    :param guess_sentences:
    :return:
    """
    correct = 0
    num_pred = 0
    num_gold = 0
    for gold, guess in zip(gold_sentences, guess_sentences):
        correct += len(set(guess.chunks) & set(gold.chunks))
        num_pred += len(guess.chunks)
        num_gold += len(gold.chunks)
    if num_pred == 0:
        prec = 0
    else:
        prec = correct/float(num_pred)
    if num_gold == 0:
        rec = 0
    else:
        rec = correct/float(num_gold)
    if prec == 0 and rec == 0:
        f1 = 0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    print("Labeled F1: " + "{0:.2f}".format(f1 * 100) +\
          ", precision: %i/%i" % (correct, num_pred) + " = " + "{0:.2f}".format(prec * 100) + \
          ", recall: %i/%i" % (correct, num_gold) + " = " + "{0:.2f}".format(rec * 100))
    return f1


# Writes labeled_sentences to outfile in the CoNLL format
def write_test_output(labeled_sentences, outfile):
    f = open(outfile, 'w')
    for sentence in labeled_sentences:
        bio_tags = sentence.get_bio_tags()
        for i in range(0, len(sentence)):
            tok = sentence.tokens[i]
            f.write(tok.word + " " + tok.pos + " " + tok.chunk + " " + bio_tags[i] + "\n")
        f.write("\n")
    print("Wrote predictions on %i labeled sentences to %s" % (len(labeled_sentences), outfile))
    f.close()