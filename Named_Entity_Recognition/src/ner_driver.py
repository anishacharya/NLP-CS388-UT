import os
import sys
sys.path.append(os.path.dirname(__file__) + '../.')

from src.data_utils.conll_reader import read_data, transform_label_for_binary_classification

from src.evaluation.ner_binary_eval import evaluate_binary_classifier, predict_binary_write_output_to_file
from src.evaluation.ner_eval import print_evaluation_metric, write_test_output

from src.classifiers.MLP_BinaryNER import train_model_based_binary_ner
from src.classifiers.label_count_driver import train_label_count_ner, train_label_count_binary_ner
from src.classifiers.hmm_ner_driver import train_hmm_ner
from src.classifiers.lstm_crf_ner_driver import train_crf_ner
from src.classifiers.emmission_crf_driver import train_emission_crf_ner
from src.classifiers.mlp_ner_driver import train_mlp_ner
import src.config as config
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
    parser.add_argument('--model', type=str, default=config.model,
                        help='model to run (Binary: COUNT, MLP; MultiClass: COUNT, HMM, CRF)')
    parser.add_argument('--mode', type=str, default=config.mode,
                        help='binary, multi_class')
    parser.add_argument('--train_path', type=str, default=config.data_path + config.language + '.train',
                        help='path to train set (you should not need to modify)')
    parser.add_argument('--dev_path', type=str, default=config.data_path + config.language + '.testa',
                        help='path to dev set')
    parser.add_argument('--blind_test_path', type=str, default=config.data_path + config.language + '.testb.blind',
                        help='path to dev set (you should not need to modify)')
    parser.add_argument('--test_output_path', type=str, default=config.output_path,
                        help='output path for test predictions')
    parser.add_argument('--no_run_on_test', dest='run_on_test', default=config.run_on_test_flag, action='store_false',
                        help='skip printing output on the test set')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    start_time = time.time()
    args = _parse_args()
    print(args)

    """ Load the training and test data """
    train_data = read_data(args.train_path)
    dev_data = read_data(args.dev_path)
    test_data = read_data(args.blind_test_path)

    if args.mode == 'binary':
        """ Convert Data into binary """
        train_class_exs = list(transform_label_for_binary_classification(train_data))
        dev_class_exs = list(transform_label_for_binary_classification(dev_data))
        test_exs = list(transform_label_for_binary_classification(test_data))

        """ Train the model """
        if args.model == "COUNT":
            classifier = train_label_count_binary_ner(train_class_exs)
        elif args.model == "MLP":
            classifier = train_model_based_binary_ner(train_class_exs)
        else:
            raise NotImplementedError("The {} model for {} mode is not implemented yet".format(args.model, args.mode))

        print("Data reading and training took %f seconds" % (time.time() - start_time))
        # Evaluate on training, development, and test data
        print("===Train accuracy===")
        evaluate_binary_classifier(train_class_exs, classifier)

        print("===Dev accuracy===")
        evaluate_binary_classifier(dev_class_exs, classifier)

        if args.run_on_test:
            print("Running on test")
            test_exs = list(transform_label_for_binary_classification(read_data(args.blind_test_path)))
            predict_binary_write_output_to_file(test_exs, classifier, args.test_output_path)
            print("Wrote predictions on %i labeled sentences to %s" % (len(test_exs), args.test_output_path))

    elif args.mode == 'multi_class':
        # Train the Model
        if args.model == "COUNT":
            model = train_label_count_ner(train_data)
            dev_decoded = [model.decode(test_ex.tokens) for test_ex in dev_data]
        elif args.model == "HMM":
            model = train_hmm_ner(train_data)
            dev_decoded = [model.decode(test_ex.tokens) for test_ex in dev_data]
        elif args.model == "CRF":
            model = train_crf_ner(train_data)
            dev_decoded = [model.decode(test_ex.tokens) for test_ex in dev_data]
        elif args.model == "ECRF":
            model = train_emission_crf_ner(train_data=train_data, dev_data=dev_data, test_data=test_data)
            dev_decoded = [model.decode(test_ex.tokens) for test_ex in dev_data]
        elif args.model == "MLP":
            model = train_mlp_ner(train_data=train_data, dev_data=dev_data, test_data=test_data)
            dev_decoded = [model.decode(test_ex.tokens) for test_ex in dev_data]
        else:
            raise NotImplementedError("The {} model for {} mode is not implemented yet".format(args.model, args.mode))

        # Print the evaluation statistics
        f1 = print_evaluation_metric(dev_data, dev_decoded)

        if args.run_on_test:
            print("Running on test")
            test = read_data(args.blind_test_path)
            test_decoded = [model.decode(test_ex.tokens) for test_ex in test]
            write_test_output(test_decoded, args.test_output_path)

    else:
        raise Exception('Only binary and multi-class mode available fix your parameter')
