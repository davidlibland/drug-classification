import pandas as pd
import numpy as np
import sqlite3 as lite
import re
from collections import Counter
import itertools

test_proportion = 0.2

class Translator():
    #input_phrases should be a list of input phrases
    #input_targets should be a corresponding list of the target classifications (also phrases)
    #min_required_word_appearances is the minimum number of times a word must appear in an input phrase 
    # for us to assume we could learn its meaning.
    def __init__(self,*args,**kwargs):
        if 'filename' in kwargs:
            self.load(kwargs['filename'])
        else:
            assert 'input_phrases' in kwargs and 'input_targets' in kwargs, "must specify a filename to load, \
                    or else give input phrase and targets"
            input_phrases = kwargs['input_phrases']
            input_targets = kwargs['input_targets']
            if 'min_required_word_appearances' in kwargs:
                self.min_required_word_appearances = kwargs['min_required_word_appearances']
            else:
                self.min_required_word_appearances = 5
            if 'max_input_phrase_length' in kwargs:
                self.max_input_phrase_length = kwargs['max_input_phrase_length']
            else:
                self.max_input_phrase_length = 32
                
            assert len(input_phrases) > self.min_required_word_appearances,"Number of input phrases %d needs to be greater than \
                        the required minimal number of word appearances %d"%(len(input_phrases),self.min_required_word_appearances)
            assert len(input_phrases) == len(input_targets),"Number of input phrases %d needs to be equal to \
                        the number of input targets %d"%(len(input_phrases),len(input_targets))
                        
            # clean the input phrases:
            cleaned_inputs = [self.clean_phrase(p) for p in input_phrases]
            # count the words, to come up with the vocabulary:
            self.vocab = Counter(itertools.chain.from_iterable(x.split(' ') for x in cleaned_inputs))
            print("Most common words are: ")
            print(self.vocab.most_common(10))
            # save the most common words:
            uncommon = []
            num_uncommon = 0
            for k, v in self.vocab.items():
                if v < self.min_required_word_appearances:
                    uncommon += [k]
                    num_uncommon += v
            for k in uncommon:
                del self.vocab[k]
            self.vocab['<unk>'] = num_uncommon #unknown words...
        
            # Create the translation dictionary from words to ids:
            self.word_to_id={word:id for id,word in enumerate(self.vocab)}
            self.id_to_word={id:word for id,word in enumerate(self.vocab)}
        
            # Create the translation dictionary from targets to ids:
            targets = list(set(input_targets))
            self.target_to_id={targ:id for id,targ in enumerate(targets)}
            self.id_to_target={id:targ for id,targ in enumerate(targets)}
        
            #turn the data into an array
            self.input_array=np.array([self.translate_clean_phrase(p) for p in cleaned_inputs])
            self.target_array=np.array([self.target_to_id[t] for t in input_targets])
            
            #drop empty data:
            mask = (self.input_array[:,0] != self.word_to_id['<eof>'])
            self.input_array = self.input_array[mask,:]
            self.target_array = self.target_array[mask]
            num_dropped = len(input_phrases)-mask.sum()
            if num_dropped>0:
                print("We dropped %d uninformative training entries, %.3f%% of the input set"
                            %(num_dropped,100.*num_dropped/len(input_phrases)))
            
            # Shuffle the data
            self.suffle_training_set()
            # Split of a testing set
            test_size = int(test_proportion*self.target_array.shape[0])
            self.input_array_test = self.input_array[:test_size,:]
            self.target_array_test = self.target_array[:test_size]
            self.input_array = self.input_array[test_size:,:]
            self.target_array = self.target_array[test_size:]
    
    #The public method that should be used to translate a phrase to ids.
    def translate_phrase(self,phrase):
        return self.translate_clean_phrase(self.clean_phrase(phrase))
        
    def translate_clean_phrase(self,phrase):
        id_list = [self.word_to_id['<unk>'] if w not in self.word_to_id else self.word_to_id[w]
                     for w in phrase.split(' ') if w != '']
        padded_list=id_list+[self.word_to_id['<eof>']]*(self.max_input_phrase_length-len(id_list))
        return padded_list
        

    #This method cleans a string, simplifying it for the RNN
    def clean_phrase(self,phrase):
        phrase = phrase.replace('\n',' ').lower()
        phrase = re.sub('[^A-Za-z0-9\-,;./\s\[\]\(\)]+', ' ', phrase)
        phrase = phrase.replace(',',' , ')\
                 .replace('.',' , ')\
                 .replace(';',' , ')\
                 .replace('(',' ( ')\
                 .replace('[',' ( ')\
                 .replace(')',' ) ')\
                 .replace(']',' ) ')\
                 .replace('-',' - ')\
                 .replace('/',' / ')
        phrase_parts = phrase.split()
        #If the string is too long, it isn't likely to be very specific
        if len(phrase_parts) > self.max_input_phrase_length-1:
            return '<eof>'
        return ' '.join(phrase_parts+['<eof>']).strip()
    
    def suffle_training_set(self):
        p = np.random.permutation(self.data_len)
        self.input_array = self.input_array[p]
        self.target_array = self.target_array[p]
        
    # This method iterates over the training/testing set.
    # it yields a pair (x,y), where x is an array of input phrases of shape (batch_size,max_input_phrase_length)
    # and y is a vector of targets of shape (batch_size).
    def id_iterator(self, batch_size,testing = False):
        self.suffle_training_set()
        if testing:
            cur_length = self.test_len
            inputs = self.input_array_test
            targets = self.target_array_test
        else:
            cur_length = self.data_len
            inputs = self.input_array
            targets = self.target_array
        
        # num_steps: int, the number of unrolls, should be a divisor of poem_length+1.
        batch_len = cur_length // batch_size

        for j in range(batch_len):
            # Load the current batch
            current_input_batch=inputs[j*batch_size:(j+1)*batch_size,:]
            current_target_batch=targets[j*batch_size:(j+1)*batch_size]
            yield (current_input_batch,current_target_batch)
            
    
    def save(self,filename):
        with open(filename,'wb') as file:
            np.savez(file, 
                 min_required_word_appearances = self.min_required_word_appearances,
                 max_input_phrase_length = self.max_input_phrase_length,
                 vocab = self.vocab, 
                 word_to_id = self.word_to_id,
                 id_to_word = self.id_to_word,
                 target_to_id = self.target_to_id,
                 id_to_target = self.id_to_target,
                 input_array = self.input_array,
                 target_array = self.target_array,
                 input_array_test = self.input_array_test,
                 target_array_test = self.target_array_test)
    
    def load(self,filename):
        with open(filename,'rb') as file:
            npzfile = np.load(file)
            self.min_required_word_appearances = npzfile['min_required_word_appearances'].item()
            self.max_input_phrase_length = npzfile['max_input_phrase_length'].item()
            self.vocab = npzfile['vocab'].item()
            self.word_to_id = npzfile['word_to_id'].item()
            self.id_to_word = npzfile['id_to_word'].item()
            self.target_to_id = npzfile['target_to_id'].item()
            self.id_to_target = npzfile['id_to_target'].item()
            self.input_array = npzfile['input_array']
            self.target_array = npzfile['target_array']
            self.input_array_test = npzfile['input_array_test']
            self.target_array_test = npzfile['target_array_test']
    
    @property
    def data_len(self):
        return self.input_array.shape[0]
        
    @property
    def test_len(self):
        return self.input_array_test.shape[0]
        