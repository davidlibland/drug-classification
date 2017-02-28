import pandas as pd
import sqlite3 as lite
import re
from collections import Counter
import itertools

class Translator():
    #input_phrases should be a list of input phrases
    #input_targets should be a corresponding list of the target classifications (also phrases)
    #min_required_word_appearances is the minimum number of times a word must appear in an input phrase 
    # for us to assume we could learn its meaning.
    def __init__(self,input_phrases,input_targets,min_required_word_appearances = 5,max_input_phrase_length = 32):
        self.max_input_phrase_length = max_input_phrase_length
        # clean the input phrases:
        cleaned_inputs = [self.clean_phrase(p) for p in input_phrases]
        # count the words, to come up with the vocabulary:
        self.vocab = Counter(itertools.chain.from_iterable(x.split(' ') for x in cleaned_inputs))
        print(self.vocab)
        # save the most common words:
        uncommon = []
        num_uncommon = 0
        for k, v in self.vocab.items():
            if v < min_required_word_appearances:
                uncommon += [k]
                num_uncommon += v
        for k in uncommon:
            del self.vocab[k]
        self.vocab['<unk>'] = num_uncommon #unknown words...
        print(self.vocab['<eof>'])
        
        # Create the translation dictionary from words to ids:
        self.word_to_id={word:id for id,word in enumerate(self.vocab)}
        self.id_to_word={id:word for id,word in enumerate(self.vocab)}
        
        # Create the translation dictionary from targets to ids:
        targets = list(set(input_targets))
        self.target_to_id={targ:id for id,targ in enumerate(targets)}
        self.id_to_target={id:targ for id,targ in enumerate(targets)}
        
        self.translated_inputs_phrases = [self.translate_clean_phrase(p) for p in cleaned_inputs]
        self.translated_targets = [self.target_to_id[t] for t in input_targets]
        
    def translate_phrase(self,phrase):
        return self.translate_clean_phrase(self.clean_phrase(phrase))
        
    def translate_clean_phrase(self,phrase):
        id_list = [self.word_to_id[w] if w in self.word_to_id else self.word_to_id['<unk>']
                     for w in phrase.split(' ') if w != '']
        padded_list=id_list+[self.word_to_id['<eof>']]*(self.max_input_phrase_length-len(id_list))
        return padded_list
        

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
        phrase=phrase+' <eof> '
        phrase_parts = phrase.split(' ')
        #If the string is too long, it isn't likely to be very specific
        if len(phrase_parts) > self.max_input_phrase_length:
            return ''
        return ' '.join(phrase_parts).strip()
    
#     def replace_uncommon_words(phrase):
#         #for w in uncommon_words:
#         my_list = phrase.split(' ')
#         my_new_list = [w if w in self.vocab else '<unk>' for w in my_list]
#         #phrase= phrase.replace(' '+w+' ',' <unk> ')
#         return ' '.join(my_new_list).strip()