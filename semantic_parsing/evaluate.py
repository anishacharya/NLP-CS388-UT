from semantic_parsing.lf_evaluator import *
from semantic_parsing.data_utils.definitions import Example
from semantic_parsing.data_utils.data_utils import print_evaluation_results
from typing import List


def evaluate(dev_data: List[Example], decoder=None, pred_derivations=None, example_freq=50, print_output=True, outfile=None):
    """
    Evaluates decoder against the data in test_data (could be dev data or test data). Prints some output
    every example_freq examples. Writes predictions to outfile if defined. Evaluation requires
    executing the model's predictions against the knowledge base. We pick the highest-scoring derivation for each
    example with a valid denotation (if you've provided more than one).
    :param pred_derivations:
    :param model:
    :param dev_data:
    :param decoder:
    :param example_freq: How often to print output
    :param print_output:
    :param outfile:
    :return:
    """
    e = GeoqueryDomain()
    if decoder is None and pred_derivations is None:
        raise NotImplementedError
    elif decoder is None:
        pred_derivations = pred_derivations
    else:
        pred_derivations = decoder.decode(dev_data)
    selected_derivs, denotation_correct = e.compare_answers([ex.y for ex in dev_data], pred_derivations, quiet=True)
    print_evaluation_results(dev_data, selected_derivs, denotation_correct, example_freq, print_output)
    # Writes to the output file if needed
    if outfile is not None:
        with open(outfile, "w") as out:
            for i, ex in enumerate(dev_data):
                out.write(ex.x + "\t" + " ".join(selected_derivs[i].y_toks) + "\n")
        out.close()
