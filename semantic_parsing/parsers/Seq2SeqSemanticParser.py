from semantic_parsing.data_utils.definitions import Example, Derivation
from typing import List


class Seq2SeqSemanticParser(object):
    def __init__(self, training_data: List[Example]):
        self.training_data = training_data

    def train(self):
        raise NotImplementedError

    def decode(self, test_data: List[Example]) -> List[List[Derivation]]:
        # Implement the inference here
        raise Exception("implement me!")
