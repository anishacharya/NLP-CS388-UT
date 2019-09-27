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


class Indexer(object):
    """
    Bijection between objects and integers starting at 0. Useful for mapping
    labels, features, etc. into coordinates of a vector space.

    Attributes:
        objs_to_ints
        ints_to_objs
    """

    def __init__(self):
        self.objs_to_ints = {}
        self.ints_to_objs = {}

    def __repr__(self):
        return str([str(self.get_object(i)) for i in range(0, len(self))])

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.objs_to_ints)

    def get_object(self, index):
        """
        :param index: integer index to look up
        :return: Returns the object corresponding to the particular index or None if not found
        """
        if index not in self.ints_to_objs:
            return None
        else:
            return self.ints_to_objs[index]

    def contains(self, object):
        """
        :param object: object to look up
        :return: Returns True if it is in the Indexer, False otherwise
        """
        return self.index_of(object) != -1

    def index_of(self, object):
        """
        :param object: object to look up
        :return: Returns -1 if the object isn't present, index otherwise
        """
        if object not in self.objs_to_ints:
            return -1
        else:
            return self.objs_to_ints[object]

    def add_and_get_index(self, object, add=True):
        """
        Adds the object to the index if it isn't present, always returns a nonnegative index
        :param object: object to look up or add
        :param add: True by default, False if we shouldn't add the object. If False, equivalent to index_of.
        :return: The index of the object
        """
        if not add:
            return self.index_of(object)

        if object not in self.objs_to_ints:
            new_idx = len(self.objs_to_ints)
            self.objs_to_ints[object] = new_idx
            self.ints_to_objs[new_idx] = object

        return self.objs_to_ints[object]


class Beam(object):
    """
    Beam data structure. Maintains a list of scored elements like a Counter, but only keeps the top n
    elements after every insertion operation. Insertion is O(n) (list is maintained in
    sorted order), access is O(1). Still fast enough for practical purposes for small beams.
    """

    def __init__(self, size):
        self.size = size
        self.elts = []
        self.scores = []

    def __repr__(self):
        return "Beam(" + repr(list(self.get_elts_and_scores())) + ")"

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.elts)

    def add(self, elt, score):
        """
        Adds the element to the beam with the given score if the beam has room or if the score
        is better than the score of the worst element currently on the beam

        :param elt: element to add
        :param score: score corresponding to the element
        """
        if len(self.elts) == self.size and score < self.scores[-1]:
            # Do nothing because this element is the worst
            return
        # If the list contains the item with a lower score, remove it
        i = 0
        while i < len(self.elts):
            if self.elts[i] == elt and score > self.scores[i]:
                del self.elts[i]
                del self.scores[i]
            i += 1
        # If the list is empty, just insert the item
        if len(self.elts) == 0:
            self.elts.insert(0, elt)
            self.scores.insert(0, score)
        # Find the insertion point with binary search
        else:
            lb = 0
            ub = len(self.scores) - 1
            # We're searching for the index of the first element with score less than score
            while lb < ub:
                m = (lb + ub) // 2
                # Check > because the list is sorted in descending order
                if self.scores[m] > score:
                    # Put the lower bound ahead of m because all elements before this are greater
                    lb = m + 1
                else:
                    # m could still be the insertion point
                    ub = m
            # lb and ub should be equal and indicate the index of the first element with score less than score.
            # Might be necessary to insert at the end of the list.
            if self.scores[lb] > score:
                self.elts.insert(lb + 1, elt)
                self.scores.insert(lb + 1, score)
            else:
                self.elts.insert(lb, elt)
                self.scores.insert(lb, score)
            # Drop and item from the beam if necessary
            if len(self.scores) > self.size:
                self.elts.pop()
                self.scores.pop()

    def get_elts(self):
        return self.elts

    def get_elts_and_scores(self):
        return zip(self.elts, self.scores)

    def head(self):
        return self.elts[0]


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
            # if label != curr_tok_label:
                # print("WARNING: invalid tag sequence (I after O); ignoring the I: %s" % bio_tags)
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










