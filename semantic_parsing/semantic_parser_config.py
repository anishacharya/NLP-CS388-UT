""" Driver args  """
data_path = '/Users/aa56927-admin/Desktop/NLP_Done_Right/semantic_parsing/data'
output_path = 'geo_test_output.tsv'

model = 'FFNN'  # RNN, FFNN
run_on_test_flag = True
run_on_manual_flag = True
seq_max_len = 60  # also can be computed more systematically looking at length distribution in corpus
model_path = './model.pt'