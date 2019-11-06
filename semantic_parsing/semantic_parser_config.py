""" Driver args  """
data_path = './data'
output_path = 'geo_test_output.tsv'

parser = 'Seq2Seq'  # 'NN'

run_on_test_flag = True
run_on_manual_flag = True
seq_max_len = 60  # also can be computed more systematically looking at length distribution in corpus
model_path = './model.pt'

epochs = 30
batch_size = 64
lr_schedule = 'None'  # None / CLR / CALR
optimizer = 'adam'  # adagrad
initial_lr = 0.01
lr_decay = 0.1
weight_decay = 1e-4
dropout = 0.2

# Stacked RNN units
no_of_rec_units = 1
# inside RNN unit
hidden_size = 100
rnn_dropout = 0.05