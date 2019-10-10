
""" Driver args  """
data_path = '/Users/anishacharya/Desktop/UT_Fall_2019/NLP/NLP_Done_Right/Sentiment_Classification/data/Rotten_Tomatoes/'
output_path = 'test-blind.output.txt'
model = 'RNN'  # RNN, FFNN
run_on_test_flag = True
seq_max_len = 60  # also can be computed more systematically looking at length distribution in corpus
model_path = './model'


if model == 'FFNN':
    #  training config
    no_classes = 2
    epochs = 5
    batch_size = 64
    lr_schedule = 'None'  # None / CLR / CALR
    optimizer = 'adam'  # adagrad
    initial_lr = 0.001
    weight_decay = 1e-4
    word_dropout_rate = 0.3

    # network config
    input_dim = 300
    hidden_1 = 150
    hidden_2 = 75
    hidden_3 = 50

    dropout = 0.2

elif model == 'RNN':
    #  training config
    no_classes = 2
    rec_unit = 'LSTM'  # GRU

    epochs = 10
    batch_size = 64
    lr_schedule = 'None'  # None / CLR / CALR
    optimizer = 'adam'  # adagrad
    initial_lr = 0.001
    lr_decay = 0.1
    weight_decay = 1e-4
    dropout = 0.2

    # Stacked RNN units
    no_of_rec_units = 2
    # inside RNN unit
    hidden_size = 50
    rnn_dropout = 0.1

""" ElMo Config """


""" BERT Config """


""" CNN Config """



