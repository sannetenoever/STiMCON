# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 10:35:16 2020

@author: sanoev
The RNN model
"""

import os
import pickle
import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding

base = ''
import RNN_subFun as seq2seq

#%% some parameters for the sentence input
SEQUENCE_LEN = 10   # amount of words in the sentence that cause a prediction
MIN_WORD_FREQUENCY = 10 # minimum amount of repetition of word to be included in the data (otherwise it is listed as an unknown word)
STEP = 1 # step in between word for the data input
BATCH_SIZE = 32 # size of the batches going into keras modeling

#%% load all the relevant things
loadloc = './Data/'
saveloc = loadloc
try:  
    os.mkdir(saveloc)
    os.mkdir(saveloc + '/checkpoints/')  
except:  
    print('checkpoint directory probably already exists')   

examples = saveloc + 'examples.txt'
meanProb = saveloc + 'meanProb.txt'           
filename = loadloc + '/WordLevel_TrainAndTestData' # the testing and training data previously created
with open(filename, 'rb') as f:
    TrainAndTest = pickle.load(f)
sentences = TrainAndTest[0] # all sentences
next_words = TrainAndTest[1] # all words to be predicted 
words = TrainAndTest[4] # a dictionary of all words

word_indices = dict((c, i) for i, c in enumerate(words)) # word to index dictionary
indices_word = dict((i, c) for i, c in enumerate(words)) # index to word dictionary
SEQUENCE_LEN = len(sentences[0]) # double check if the sequence length is good

filename = loadloc + '/WordLevel_EmbWeights'  # these are the embedding weights
with open(filename, 'rb') as f:
    LayerInf = pickle.load(f)
embedding_matrix = LayerInf[0]

#%% the model
# split testing and training
(sentences, next_words), (sentences_test, next_words_test), (DNNindex, cutindex) = seq2seq.shuffle_and_split_training_set(
    sentences, next_words, percentage_test = 5)
# calculate perplexity (based on training only!)
NLTKmodel = seq2seq.cal_perplexity_model(sentences, next_words)

# save this, so you can also run stuff just ont he validation set:
#filename = saveloc + '/TrainAndTestSplits'            
#with open(filename, 'wb') as f:
#    pickle.dump([sentences, next_words, sentences_test, next_words_test, DNNindex, cutindex],f)  

#%% load splits
filename = loadloc + '/TrainAndTestSplits'            
with open(filename, 'rb') as f:
    allsplits = pickle.load(f)  
sentences = allsplits[0]
next_words = allsplits[1]
sentences_test = allsplits[2]
next_words_test = allsplits[3]
DNNindex = allsplits[4]
cutindex = allsplits[5]

#%% make the model
print('Build model...')
model = Sequential()
model.add(Embedding(input_dim=len(words),output_dim=len(embedding_matrix[0])))
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False
model.add(LSTM(300, return_sequences = False, recurrent_dropout = 0.2, dropout = 0.2, activation = 'tanh'))
#model.add(Dense(len(words), kernel_regularizer=regularizers.l2(0.01), activation = 'softmax'))
model.add(Dense(len(words), activation = 'softmax'))
opt = keras.optimizers.Adam(learning_rate = 0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['acc'])
model.summary()

file_path = saveloc + "/checkpoints/LSTM-epoch{epoch:03d}-words%d-sequence%d-minfreq%d-" \
            "loss{loss:.4f}-acc{acc:.4f}-val_loss{val_loss:.4f}-val_acc{val_acc:.4f}" % \
            (len(words), SEQUENCE_LEN, MIN_WORD_FREQUENCY)

examples_file = open(examples, "w")
meanProb_file = open(meanProb, "w")

checkpoint = ModelCheckpoint(file_path, monitor='val_acc', save_best_only=False)
#early_stopping = EarlyStopping(monitor='val_acc', patience=100)
callbacks_list = [checkpoint, seq2seq.CustomModelEval(examples_file, meanProb_file, SEQUENCE_LEN, word_indices, 
                                                      indices_word, sentences, next_words, sentences_test, next_words_test,NLTKmodel)]

model.fit_generator(seq2seq.generator(sentences, next_words, BATCH_SIZE, SEQUENCE_LEN, word_indices),
                    steps_per_epoch=int(len(sentences)/BATCH_SIZE) + 1,
                    epochs=100,
                    callbacks=callbacks_list,
                    validation_data=seq2seq.generator(sentences_test, next_words_test, BATCH_SIZE, SEQUENCE_LEN, word_indices),
                    validation_steps=int(len(sentences_test)/BATCH_SIZE) + 1)

