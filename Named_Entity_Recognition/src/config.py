""" Driver args -> specify model, mode, file paths etc"""
data_path = '/Users/anishacharya/Desktop/UT_Fall_2019/NLP/NLP_Done_Right/Named_Entity_Recognition/data/CONLL_2003/'
language = 'eng'  # eng, du

model = 'HMM'  # (Binary: COUNT, MLP; MultiClass: COUNT, HMM, CRF)'
mode = 'multi_class'  # binary, multi_class
output_path = 'eng.testb.out'

""" define model hyper-parameters here """
glove_file = '/Users/anishacharya/Desktop/glove.6B/glove.6B.300d.txt'
epochs = 1
batch_size = 64
initial_lr = 0.01
no_of_classes = 2