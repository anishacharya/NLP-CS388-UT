""" Driver args  """
data_path = './data'
output_path = 'geo_test_output.tsv'

parser = 'Seq2Seq'  # 'NN'

run_on_test_flag = True
run_on_manual_flag = True
seq_max_len = 60  # also can be computed more systematically looking at length distribution in corpus
model_path = './model.pt'

epochs = 10
batch_size = 64
lr_schedule = 'None'  # None / CLR / CALR
optimizer = 'adam'  # adagrad
initial_lr = 0.001
lr_decay = 0.1
weight_decay = 1e-4
dropout = 0.2

# Stacked RNN units
no_of_rec_units = 1

# inside RNN unit
enc_hidden_size = 50
enc_rnn_dropout = 0.3

dec_embed_dim = 50
dec_hidden_size = 50
dec_rnn_dropout = 0.3

teacher_force_train = 0.6
teacher_force_test = 0.0
