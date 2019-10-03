import sklearn.metrics as metrics
import numpy as np


class ClassificationEval:
    def __init__(self, ground_truth: np.array, prediction: np.array):
        self.ground_truth = ground_truth
        self.prediction = prediction

        self.accuracy = self.accuracy()
        self.precision = self.precision_micro()
        self.recall = self.recall_micro()
        self.f1_score = 2 * (self.precision * self.recall)/(self.precision + self.recall)

    def accuracy(self) -> float:
        """
        fraction of correct predictions , set normalize = false
        if we need number of correct sample instead of fraction
        """
        accuracy = metrics.accuracy_score(self.ground_truth, self.prediction, normalize=True)
        return accuracy

    def recall_micro(self):
        """Calculate metrics for each label, and find their unweighted mean.
        This does not take label imbalance into account.
        """
        micro_recall = metrics.recall_score(self.ground_truth, self.prediction, average='micro')
        return micro_recall

    def precision_micro(self):
        """
        Calculate metrics globally by counting the total true positives,
        false negatives and false positives.
        """
        micro_precision = metrics.precision_score(self.ground_truth, self.prediction, average='micro')
        return micro_precision
