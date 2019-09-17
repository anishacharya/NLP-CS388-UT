"""
Author: Anish Acharya <anishacharya@utexas.edu>
Adopted From: Greg Durret <gdurrett@cs.utexas.edu>
"""

from typing import List


class Token:
    """
    Abstraction to bundle words with POS and syntactic chunks for featurization

    Attributes:
        word: string
        pos: string part-of-speech
        chunk: string representation of the syntactic chunk (e.g., I-NP). These can be useful
        features but you don't need to use them.
    """
    def __init__(self, word: str, pos: str, chunk: str):
        self.word = word
        self.pos = pos
        self.chunk = chunk

    def __repr__(self):
        return "Token(%s, %s, %s)" % (self.word, self.pos, self.chunk)

    def __str__(self):
        return self.__repr__()


class Chunk:
    """
    Thin wrapper around a start and end index coupled with a label, representing,
    e.g., a chunk PER over the span (3,5). Indices are semi-inclusive, so (3,5)
    contains tokens 3 and 4 (0-based indexing).

    Attributes:
        start_idx:
        end_idx:
        label: str
    """
    def __init__(self, start_idx: int, end_idx: int, label: str):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.label = label

    def __repr__(self):
        return "(" + repr(self.start_idx) + ", " + repr(self.end_idx) + ", " + self.label + ")"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.start_idx == other.start_idx and self.end_idx == other.end_idx and self.label == other.label
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.start_idx) + hash(self.end_idx) + hash(self.label)


class LabeledSentence:
    """
    Thin wrapper over a sequence of Tokens representing a sentence and an optional set of chunks
    representation NER labels, which are also stored as BIO tags

    Attributes:
        tokens: list[Token]
        chunks: list[Chunk]
        bio_tags: list[str]
    """
    def __init__(self, tokens: List[Token], chunks: List[Chunk]):
        self.tokens = tokens
        self.chunks = chunks
        if chunks is None:
            self.bio_tags = None
        else:
            self.bio_tags = bio_tags_from_chunks(self.chunks, len(self.tokens))

    def __repr__(self):
        return repr([repr(tok) for tok in self.tokens]) + "\n" + repr([repr(chunk) for chunk in self.chunks])

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.tokens)

    def get_bio_tags(self):
        return self.bio_tags


class PersonExample(object):
    """
    Data wrapper for a single sentence for person classification, which consists of many individual tokens to classify.

    Attributes:
        tokens: the sentence to classify : A list of Token Objects
        labels: 0 if non-person name, 1 if person name for each token in the sentence
    """
    def __init__(self, tokens: List[Token],
                 labels: List[int]):
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.tokens)


def isB(ner_tag: str):
    """
    We store NER tags as strings, but they contain two pieces: a coarse tag type (BIO) and a label (PER), e.g. B-PER
    :param ner_tag:
    :return:
    """
    return ner_tag.startswith("B")


def isI(ner_tag: str):
    return ner_tag.startswith("I")


def isO(ner_tag: str):
    return ner_tag == "O"


def get_tag_label(ner_tag: str):
    """
    :param ner_tag:
    :return: The label component of the NER tag: e.g., returns PER for B-PER
    """
    if len(ner_tag) > 2:
        return ner_tag[2:]
    else:
        return None


def chunks_from_bio_tag_seq(bio_tags: List[str]) -> List[Chunk]:
    """
    Convert BIO tags to (start, end, label) chunk representations.
    O   O  B-PER  I-PER     O
    He met Barack Obama yesterday
    => [Chunk(2, 4, PER)]
    N.B. this method only works because chunks are non-overlapping in this data
    :param bio_tags: list[str] of BIO tags
    :return: list[Chunk] encodings of the NER chunks
    """
    chunks = []
    curr_tok_start = -1
    curr_tok_label = ""
    for idx, tag in enumerate(bio_tags):
        if isB(tag):
            label = get_tag_label(tag)
            if curr_tok_label != "":
                chunks.append(Chunk(curr_tok_start, idx, curr_tok_label))
            curr_tok_label = label
            curr_tok_start = idx
        elif isI(tag):
            label = get_tag_label(tag)
            if label != curr_tok_label:
                print("WARNING: invalid tag sequence (I after O); ignoring the I: %s" % bio_tags)
        else: # isO(tag):
            if curr_tok_label != "":
                chunks.append(Chunk(curr_tok_start, idx, curr_tok_label))
            curr_tok_label = ""
            curr_tok_start = -1
    # If the sentence ended in the middle of a tag
    if curr_tok_start >= 0:
        chunks.append(Chunk(curr_tok_start, len(bio_tags), curr_tok_label))
    return chunks


def bio_tags_from_chunks(chunks: List[Chunk], sent_len: int) -> List[str]:
    """
    Converts a chunk representation back to BIO tags
    :param chunks:
    :param sent_len:
    :return:
    """
    tags = []
    for i in range(0, sent_len):
        matching_chunks = list(filter(lambda chunk: chunk.start_idx <= i < chunk.end_idx, chunks))
        if len(matching_chunks) > 0:
            if i == matching_chunks[0].start_idx:
                tags.append("B-" + matching_chunks[0].label)
            else:
                tags.append("I-" + matching_chunks[0].label)
        else:
            tags.append("O")
    return tags









