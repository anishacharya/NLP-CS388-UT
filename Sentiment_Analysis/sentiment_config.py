
""" Driver args  """
data_path = '/Users/anishacharya/Desktop/UT_Fall_2019/NLP/NLP_Done_Right/Sentiment_Analysis/data/Rotten_Tomatoes/'
output_path = 'test-blind.output.txt'
model = 'FFNN'  # LSTM
run_on_test_flag = True
seq_max_len = 60  # also can be computed more systematically looking at length distribution in corpus

""" FFNN config """
#  training config
ffnn_epochs = 15
batch_size = 64
lr_schedule = 'None'  # None / CLR / CALR
optimizer = 'adam'  # adagrad
initial_lr = 0.01
weight_decay = 1e-4

word_dropout_rate = 0.2
no_classes = 2
# network config


""" LSTM config """






