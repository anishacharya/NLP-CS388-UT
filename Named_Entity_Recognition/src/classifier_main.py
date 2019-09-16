from src.evaluation.ner_eval import *
from src.data_utils.nerdata import *
from src.classifiers.fnn import train_model_based_ner
from src.classifiers.label_count_classifier import train_count_based_binary_ner
import argparse
import time


"""
Author: Anish Acharya <anishacharya@utexas.edu>
Adopted From: Greg Durret <gdurrett@cs.utexas.edu>
"""


def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--model', type=str, default='FNN', help='model to run (COUNT, FNN)')
    parser.add_argument('--mode', type=str, default='multiclass', help='binary, multiclass')
    parser.add_argument('--train_path', type=str, default='data/CONLL_2003/eng.train',
                        help='path to train set (you should not need to modify)')
    parser.add_argument('--dev_path', type=str, default='data/CONLL_2003/eng.testa',
                        help='path to dev set (you should not need to modify)')
    parser.add_argument('--blind_test_path', type=str, default='data/CONLL_2003/eng.testb.blind',
                        help='path to dev set (you should not need to modify)')
    parser.add_argument('--test_output_path', type=str, default='eng.testb.out',
                        help='output path for test predictions')
    parser.add_argument('--no_run_on_test', dest='run_on_test', default=True, action='store_false',
                        help='skip printing output on the test set')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    start_time = time.time()
    args = _parse_args()
    print(args)

    # Load the training and test data
    train_class_exs = list(transform_label_for_binary_classification(read_data(args.train_path)))
    dev_class_exs = list(transform_label_for_binary_classification(read_data(args.dev_path)))
    test_exs = list(transform_label_for_binary_classification(read_data(args.blind_test_path)))

    # Train the model
    if args.model == "COUNT":
        classifier = train_count_based_binary_ner(train_class_exs)
    else:
        classifier = train_model_based_ner(train_class_exs)

    print("Data reading and training took %f seconds" % (time.time() - start_time))
    # Evaluate on training, development, and test data
    print("===Train accuracy===")
    evaluate_classifier(train_class_exs, classifier)

    print("===Dev accuracy===")
    evaluate_classifier(dev_class_exs, classifier)

    if args.run_on_test:
        print("Running on test")
        test_exs = list(transform_label_for_binary_classification(read_data(args.blind_test_path)))
        predict_write_output_to_file(test_exs, classifier, args.test_output_path)
        print("Wrote predictions on %i labeled sentences to %s" % (len(test_exs), args.test_output_path))



