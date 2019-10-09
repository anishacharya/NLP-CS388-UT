from Sentiment_Classification.src.data_utils.definitions import SentimentExample
from common.utils.utils import TextCleaning
from common.utils.indexer import Indexer
import common.common_config as common_conf
from typing import List


def read_and_index_sentiment_examples(infile: str,
                                      word_indexer: Indexer,
                                      add_to_indexer=False,
                                      word_counter=None) -> List[SentimentExample]:
    """
    Reads sentiment examples in the format [0 or 1]<TAB>[raw sentence]; tokenizes and indexes the sentence according
    to the vocabulary in indexer.
    :param infile: file to read
    :param word_indexer: Indexer containing the vocabulary
    :param add_to_indexer: If add_to_indexer is False, replaces unseen words with UNK, otherwise grows the indexer.
    :param word_counter: optionally keeps a tally of how many times each word is seen (mostly for logging purposes).
    :return: A list of SentimentExample objects read from the file
    """
    # f = open(infile, encoding='utf-8')
    f = open(infile, encoding='iso8859')
    # initialize a Text Cleaner
    text_cleaner = TextCleaning()
    sentiment_data = []
    word_indexer.add_and_get_index(common_conf.UNK_TOKEN)
    word_indexer.add_and_get_index(common_conf.PAD_TOKEN)

    for line in f:
        if len(line.strip()) > 0:
            fields = line.split("\t")

            label = int(fields[0])
            sent = fields[1]

            tokenized_cleaned_sent = list(filter(lambda x: x != '', text_cleaner.text_cleaning(sent)))

            if word_counter is not None:
                for word in tokenized_cleaned_sent:
                    word_counter[word] += 1.0

            indexed_sent = [word_indexer.add_and_get_index(word) if word_indexer.contains(word) or add_to_indexer
                            else word_indexer.index_of(common_conf.UNK_TOKEN) for word in tokenized_cleaned_sent]
            sentiment_data.append(SentimentExample(indexed_sent, label))

    f.close()
    return sentiment_data


def write_sentiment_examples(exs: List[SentimentExample],
                             outfile: str,
                             indexer: Indexer):
    """
    Writes sentiment examples to an output file in the same format they are read in. Note that what gets written
    out is tokenized, so this will not exactly match the input file. However, this is fine from the standpoint of
    writing model output.
    """
    o = open(outfile, 'w')
    for ex in exs:
        o.write(repr(ex.label) + "\t" + " ".join([indexer.get_object(idx) for idx in ex.indexed_words]) + "\n")
    o.close()
