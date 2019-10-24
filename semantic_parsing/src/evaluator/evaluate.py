from semantic_parsing.src.evaluator.lf_evaluator import *
from typing import List


def evaluate(test_data: List[Example], decoder, example_freq=50, print_output=True, outfile=None):
    """
    Evaluates decoder against the data in test_data (could be dev data or test data). Prints some output
    every example_freq examples. Writes predictions to outfile if defined. Evaluation requires
    executing the model's predictions against the knowledge base. We pick the highest-scoring derivation for each
    example with a valid denotation (if you've provided more than one).
    :param test_data:
    :param decoder:
    :param example_freq: How often to print output
    :param print_output:
    :param outfile:
    :return:
    """
    e = GeoqueryDomain()
    pred_derivations = decoder.decode(test_data)
    selected_derivs, denotation_correct = e.compare_answers([ex.y for ex in test_data], pred_derivations)
    print_evaluation_results(test_data, selected_derivs, denotation_correct, example_freq, print_output)
    # Writes to the output file if needed
    if outfile is not None:
        with open(outfile, "w") as out:
            for i, ex in enumerate(test_data):
                out.write(ex.x + "\t" + " ".join(selected_derivs[i].y_toks) + "\n")
        out.close()
