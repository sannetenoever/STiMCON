#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 10:41:49 2020

@author: sanoev
Subfunction to use the RNN_Model
"""

import math
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional, Embedding
import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from nltk.lm import Vocabulary

#%%
def shuffle_and_split_training_set(sentences_original, next_original, percentage_test=2):
    # shuffle at unison
    print('Shuffling sentences')

    tmp_sentences = []; tmp_sentences_inx = [];
    tmp_next_word = [];
    for i in np.random.permutation(len(sentences_original)):
        tmp_sentences_inx.append(i)
        tmp_sentences.append(sentences_original[i])
        tmp_next_word.append(next_original[i])

    cut_index = int(len(sentences_original) * (1.-(percentage_test/100.)))
    x_train, x_test = tmp_sentences[:cut_index], tmp_sentences[cut_index:]
    y_train, y_test = tmp_next_word[:cut_index], tmp_next_word[cut_index:]

    print("Size of training set = %d" % len(x_train))
    print("Size of test set = %d" % len(y_test))
    return (x_train, y_train), (x_test, y_test), (tmp_sentences_inx, cut_index)


# Data generator for fit and evaluate
def generator(sentence_list, next_word_list, batch_size, SEQUENCE_LEN, word_indices):
    index = 0
    while True:
        x = np.zeros((batch_size, SEQUENCE_LEN), dtype=np.int32)
        y = np.zeros((batch_size), dtype=np.int32)
        for i in range(batch_size):
            for t, w in enumerate(sentence_list[index % len(sentence_list)]):
                x[i, t] = word_indices[w]
            y[i] = word_indices[next_word_list[index % len(sentence_list)]]
            index = index + 1
        yield x, y

# Data generator for fit and evaluate
def generator_return(sentence_list, next_word_list, batch_size, SEQUENCE_LEN, word_indices):
    index = 0
    x = np.zeros((batch_size, SEQUENCE_LEN), dtype=np.int32)
    y = np.zeros((batch_size), dtype=np.int32)
    for i in range(batch_size):
        for t, w in enumerate(sentence_list[index % len(sentence_list)]):
            x[i, t] = word_indices[w]
        y[i] = word_indices[next_word_list[index % len(sentence_list)]]
        index = index + 1
    return x, y

def cal_perplexity_model(sentences, next_words, Ngram = 0):
    # https://stackoverflow.com/questions/54941966/how-can-i-calculate-perplexity-using-nltk   
    n=2    
    tokenized_text = [sen + [next_words[it]] for it,sen in enumerate(sentences)]
    train_data = [nltk.bigrams(t,  pad_right=False, pad_left=False) for t in tokenized_text]
    words = [word for sent in tokenized_text for word in sent]
    padded_vocab = Vocabulary(words)
    NLTKmodel = MLE(n)
    NLTKmodel.fit(train_data, padded_vocab)
    return NLTKmodel

def get_model(words, SEQUENCE_LEN = [], dropout=0.2, EmbLay = []):
    print('Build model...')
    model = Sequential()
    if len(EmbLay) == 0:
        model.add(Embedding(input_dim=len(words), output_dim=300))
    else:       
        model.add(Embedding(input_dim=len(words), output_dim = len(EmbLay[0])))
        model.layers[0].set_weights([EmbLay])
        model.layers[0].trainable = False
    model.add(Bidirectional(LSTM(128)))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(len(words)))
    model.add(Activation('softmax'))
    return model


# Functions from keras-team/keras/blob/master/examples/lstm_text_generation.py
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

#%% on ending of an epoch
# define a class for the model
class CustomModelEval(keras.callbacks.Callback):
    def __init__(self, examples_file, meanProb_file, SEQUENCE_LEN, word_indices, indices_word, sentences, next_words, sentences_test, next_words_test, bigramModel):
        self.sentences = sentences
        self.next_words = next_words
        self.sentences_test = sentences_test
        self.next_words_test= next_words_test
        self.examples_file = examples_file
        self.meanProb_file = meanProb_file
        self.SEQUENCE_LEN = SEQUENCE_LEN
        self.word_indices = word_indices
        self.indices_word = indices_word
        self.bigram = bigramModel

    def on_epoch_end(self,epoch, logs):
        # Function invoked at end of each epoch. Prints generated text.
        self.examples_file.write('\n----- Generating text after Epoch: %d\n' % epoch)
    
        # Randomly pick a seed sequence
        seed_index = np.random.randint(len(self.sentences+self.sentences_test))
        seed = (self.sentences+self.sentences_test)[seed_index]
    
        for diversity in [0.3, 0.4, 0.5, 0.6, 0.7]:
            sentence = seed
            self.examples_file.write('----- Diversity:' + str(diversity) + '\n')
            self.examples_file.write('----- Generating with seed:\n"' + ' '.join(sentence) + '"\n')
            self.examples_file.write(' '.join(sentence))
    
            for i in range(50):
                x_pred = np.zeros((1, self.SEQUENCE_LEN))
                for t, word in enumerate(sentence):
                    x_pred[0, t] = self.word_indices[word]
    
                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_word = self.indices_word[next_index]
    
                sentence = sentence[1:]
                sentence.append(next_word)
    
                self.examples_file.write(" "+next_word)
            self.examples_file.write('\n')
        self.examples_file.write('='*80 + '\n')
        self.examples_file.flush()
        
        # get the mean probability and perplexity of the model:
        # do in many batches...
        bs = 10
        stepT = int(len(self.sentences)/bs)
        AP = np.array([])
        indMaxWord = np.array([])
        for bv in range(4):            
            rangeV = [stepT*bv,stepT*(bv+1)]
            add = stepT*bv
            if bv == 3:
                rangeV[1]=len(self.sentences)+1
               
            # probability
            [modelInput,labels] = generator_return(self.sentences[rangeV[0]:rangeV[1]], self.next_words[rangeV[0]:rangeV[1]], len(self.sentences[rangeV[0]:rangeV[1]]), 10, self.word_indices)
            preds = self.model.predict(modelInput)     
            indMaxWord=np.append(indMaxWord, np.argmax(preds,1))
            for it in range(preds.shape[0]):
                AP = np.append(AP, preds[:,labels[it+add]])
                      
        # perplexity
        tokenized_text = [sen + [self.indices_word[indMaxWord[it]]] for it,sen in enumerate(self.sentences)] # the whole sentence        
        test_data = [nltk.bigrams(t,  pad_right=False, pad_left=False) for t in tokenized_text]
        tokenized_text_bi = [[sen[-1]] + [self.indices_word[indMaxWord[it]]] for it,sen in enumerate(self.sentences)] # the whole sentence        
        test_data_bigram = [nltk.bigrams(t,  pad_right=False, pad_left=False) for t in tokenized_text_bi]
         
        adpr = 0; adper = 0; adbi = 0; cntper = 0; cntbi = 0
        for it in range(len(self.sentences)):           
            #adpr = adpr+preds[it,labels[it]] # probability actual output
            adpr = adpr+AP[it]
            per = self.bigram.perplexity(test_data[it])
            bi = self.bigram.perplexity(test_data_bigram[it])
            if math.isinf(per) == False:
                adper = adper + per
                cntper = cntper+1
            if math.isinf(bi) == False:
                adbi = adbi + bi
                cntbi = cntbi+1
        meanP = adpr/len(self.next_words)
        meanPer = adper/cntper
        meanBi= adbi/cntbi
                     
        # validation data
        # probability
        [modelInput,labels] = generator_return(self.sentences_test, self.next_words_test, len(self.sentences_test), 10, self.word_indices)
        preds = self.model.predict(modelInput)
        # perplexity
        indMaxWord = np.argmax(preds,1)
        tokenized_text = [sen + [self.indices_word[indMaxWord[it]]] for it,sen in enumerate(self.sentences_test)] # the whole sentence        
        test_data = [nltk.bigrams(t,  pad_right=False, pad_left=False) for t in tokenized_text]
        tokenized_text_bi = [[sen[-1]] + [self.indices_word[indMaxWord[it]]] for it,sen in enumerate(self.sentences_test)] # the whole sentence        
        test_data_bigram = [nltk.bigrams(t,  pad_right=False, pad_left=False) for t in tokenized_text_bi]
        
        adpr = 0; adper = 0; adbi = 0; cntperVal = 0; cntbiVal = 0
        for it in range(len(self.sentences_test)):           
            adpr = adpr+preds[it,labels[it]] # probability actual output
            per = self.bigram.perplexity(test_data[it])
            bi = self.bigram.perplexity(test_data_bigram[it])
            if math.isinf(per) == False:
                adper = adper + per
                cntperVal = cntperVal+1
            if math.isinf(bi) == False:
                adbi = adbi + bi
                cntbiVal = cntbiVal+1
        meanPVal = adpr/len(self.next_words_test)
        meanPerVal = adper/cntperVal
        meanBiVal = adbi/cntbiVal
       
        self.meanProb_file.write('\n----- Prop/Per/Bigram after Epoch: %d\n' % epoch)
        self.meanProb_file.write('PropTest\t' + str(meanP) + '\t' + str(len(self.next_words)) + '\n')
        self.meanProb_file.write('PropVal\t' + str(meanPVal) + '\t' + str(len(self.next_words_test)) +  '\n')
        self.meanProb_file.write('PerplexicityTest\t' + str(meanPer) + '\t' + str(cntper/len(self.next_words)) + '\n')
        self.meanProb_file.write('PerplexityVal\t' + str(meanPerVal) + '\t' + str(cntperVal/len(self.next_words_test)) + '\n')
        self.meanProb_file.write('BigramTest\t' + str(meanBi) + '\t' + str(cntbi/len(self.next_words)) + '\n')
        self.meanProb_file.write('BigramVal\t' + str(meanBiVal) + '\t' + str(cntbiVal/len(self.next_words_test)) +  '\n')
        self.meanProb_file.flush()
    
#%% on ending of an epoch
# define a class for the model
class CustomModelEval_v2(keras.callbacks.Callback):
    def __init__(self, examples_file, meanProb_file, SEQUENCE_LEN, word_indices, indices_word, sentences, next_words, sentences_test, next_words_test, bigramModel):
        self.sentences = sentences
        self.next_words = next_words
        self.sentences_test = sentences_test
        self.next_words_test= next_words_test
        self.examples_file = examples_file
        self.meanProb_file = meanProb_file
        self.SEQUENCE_LEN = SEQUENCE_LEN
        self.word_indices = word_indices
        self.indices_word = indices_word
        self.bigram = bigramModel

    def on_epoch_end(self,epoch, logs):
        # Function invoked at end of each epoch. Prints generated text.
        self.examples_file.write('\n----- Generating text after Epoch: %d\n' % epoch)
    
        # Randomly pick a seed sequence
        seed_index = np.random.randint(len(self.sentences+self.sentences_test))
        seed = (self.sentences+self.sentences_test)[seed_index]
    
        for diversity in [0.3, 0.4, 0.5, 0.6, 0.7]:
            sentence = seed
            self.examples_file.write('----- Diversity:' + str(diversity) + '\n')
            self.examples_file.write('----- Generating with seed:\n"' + ' '.join(sentence) + '"\n')
            self.examples_file.write(' '.join(sentence))
    
            for i in range(50):
                x_pred = np.zeros((1, self.SEQUENCE_LEN))
                for t, word in enumerate(sentence):
                    x_pred[0, t] = self.word_indices[word]
    
                preds = self.model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_word = self.indices_word[next_index]
    
                sentence = sentence[1:]
                sentence.append(next_word)
    
                self.examples_file.write(" "+next_word)
            self.examples_file.write('\n')
        self.examples_file.write('='*80 + '\n')
        self.examples_file.flush()
        
        # get the mean probability and perplexity of the model (only of valdation).. Training takes too long:
        # do in many batches...
    
                     
        # validation data
        # probability
        [modelInput,labels] = generator_return(self.sentences_test, self.next_words_test, len(self.sentences_test), 10, self.word_indices)
        preds = self.model.predict(modelInput)
        # perplexity
        indMaxWord = np.argmax(preds,1)
        tokenized_text = [sen + [self.indices_word[indMaxWord[it]]] for it,sen in enumerate(self.sentences_test)] # the whole sentence        
        test_data = [nltk.bigrams(t,  pad_right=False, pad_left=False) for t in tokenized_text]
        tokenized_text_bi = [[sen[-1]] + [self.indices_word[indMaxWord[it]]] for it,sen in enumerate(self.sentences_test)] # the whole sentence        
        test_data_bigram = [nltk.bigrams(t,  pad_right=False, pad_left=False) for t in tokenized_text_bi]
        
        adpr = 0; adper = 0; adbi = 0; cntperVal = 0; cntbiVal = 0
        for it in range(len(self.sentences_test)):           
            adpr = adpr+preds[it,labels[it]] # probability actual output
            per = self.bigram.perplexity(test_data[it])
            bi = self.bigram.perplexity(test_data_bigram[it])
            if math.isinf(per) == False:
                adper = adper + per
                cntperVal = cntperVal+1
            if math.isinf(bi) == False:
                adbi = adbi + bi
                cntbiVal = cntbiVal+1
        meanPVal = adpr/len(self.next_words_test)
        meanPerVal = adper/cntperVal
        meanBiVal = adbi/cntbiVal
        cntper = 0
        meanP = 0
        meanPer = 0
        meanBi = 0
        cntbi = 0
        
       
        self.meanProb_file.write('\n----- Prop/Per/Bigram after Epoch: %d\n' % epoch)
        self.meanProb_file.write('PropTest\t' + str(meanP) + '\t' + str(len(self.next_words)) + '\n')
        self.meanProb_file.write('PropVal\t' + str(meanPVal) + '\t' + str(len(self.next_words_test)) +  '\n')
        self.meanProb_file.write('PerplexicityTest\t' + str(meanPer) + '\t' + str(cntper/len(self.next_words)) + '\n')
        self.meanProb_file.write('PerplexityVal\t' + str(meanPerVal) + '\t' + str(cntperVal/len(self.next_words_test)) + '\n')
        self.meanProb_file.write('BigramTest\t' + str(meanBi) + '\t' + str(cntbi/len(self.next_words)) + '\n')
        self.meanProb_file.write('BigramVal\t' + str(meanBiVal) + '\t' + str(cntbiVal/len(self.next_words_test)) + '\n')
        self.meanProb_file.flush()