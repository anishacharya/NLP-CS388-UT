""" Driver args -> specify model, mode, file paths etc"""
data_path = '/Users/anishacharya/Desktop/UT_Fall_2019/NLP/NLP_Done_Right/Named_Entity_Recognition/data/CONLL_2003/'
language = 'eng'  # eng, du

model = 'CRF'  # (Binary: COUNT, MLP; MultiClass: COUNT, HMM, CRF)'
mode = 'multi_class'  # binary, multi_class
output_path = 'eng.testb.out'
run_on_test_flag = True

""" define model hyper-parameters here """
glove_file = '/Users/anishacharya/Desktop/glove.6B/glove.6B.300d.txt'
epochs = 15
batch_size = 64
initial_lr = 0.01
weight_decay = 1e-4
no_of_classes = 2
embedding_dim = 50
hidden_dim = 10

# define special symbols to handle padding etc
UNK_TOKEN = "__UNK__"
PAD_TOKEN = "__PAD__"
BOS_TOKEN = "__BOS__"
EOS_TOKEN = "__EOS__"
