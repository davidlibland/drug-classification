#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from .model import RNNClassifierArgs, RNNClassifierModel
from . import config
from .translator import Translator
import os
import dill

class TrainedClassifier():
    def __init__(self,translator_filename,save_dir=os.path.join(config.base_dir,config.sampling_model_dir)):
        print("-"*30)
        print("loading the translator")
        print("-"*30)
        self.translatorObj = Translator(filename=translator_filename)
        
        self.sess = tf.Session()
        self.load_model(save_dir)

    def load_model(self,save_dir=os.path.join(config.base_dir,config.sampling_model_dir)):
    # Load the saved dictionary
        print("-"*30)
        print("loading the args")
        print("-"*30)
        with open(os.path.join(save_dir,config.args_file),'rb') as f:
            dict_data=dill.load(f)
            self.args=RNNClassifierArgs()
            self.args.__dict__=dict_data['args'].__dict__
        # Load the saved model
        print("-"*30)
        print("initialization")
        print("-"*30)
        #self.args.batch_size=1
        self.args.log_dir=save_dir   
    
        if self.sess==None:
            self.sess=tf.Session()
    
        # Initializer
        initializer = tf.random_uniform_initializer(-self.args.init_scale,self.args.init_scale)
        with tf.variable_scope("model", reuse = None, initializer = initializer):
            self.p_classifier = RNNClassifierModel(args=self.args, is_training = False)
                              
    
        # Initialize the variables
        self.sess.run(tf.initialize_all_variables())
        
        all_vars = tf.all_variables()
        model_vars = [k for k in all_vars if k.name.startswith("model")]

        saver = tf.train.Saver(model_vars)


        print("-"*30)
        print("loading the model")
        print("-"*30)
        saver.restore(self.sess, os.path.join(save_dir,config.sampling_model))
    

    # phrases is a list of N (dirty) strings on which to run the classifier
    # returns a list of N dicts assigning confidences to the various classes.
    def confidence_classification_from_active_sess(self,phrases):
#         print("-"*30)
#         print("Running the classifier")
#         print("-"*30)
        p = self.p_classifier
        
        out_probs = []
        for i,x in enumerate(self.translatorObj.translating_iterator(phrases,self.args.batch_size)):
            #state = self.sess.run(p.initial_state)
            cur_probs, state, _ = self.sess.run([p.output_prob, p.final_state,p.initial_state],
                                     {p.input_IDs: x})
            out_probs+=[{t:cur_probs[j,k] for k,t in self.translatorObj.id_to_target.items()}
                     for j in range(cur_probs.shape[0])]
        return out_probs[:len(phrases)]
    
    # phrases is a list of N (dirty) strings on which to run the classifier
    # returns a list of N proposed classifications
    def top_classification_from_active_sess(self,phrases):
        out_probs = self.confidence_classification_from_active_sess(phrases)
        return [max(stats, key=stats.get) for stats in out_probs]
    
    # phrases is a list of N (dirty) strings WHICH ALL CORRESPOND TO THE SAME COMPOUND
    # on which to run the classifier
    # returns a single dict assigning confidences to the various classes.
    def confidence_classification_aggregate(self,phrases):
        probs = self.confidence_classification_from_active_sess(phrases)
        count = len(probs)
        if count == 0:
            return {t:1./len(self.translatorObj.id_to_target.items()) for k,t in self.translatorObj.id_to_target.items()}
        
        agg_probs = {key:0 for key in probs[0].keys()}
        for prob in probs:
            for key in agg_probs.keys():
                agg_probs[key] += prob[key]
        
        for key in agg_probs.keys():
            agg_probs[key] /= count
        
        return agg_probs
        
    
    # phrases is a list of N (dirty) strings WHICH ALL CORRESPOND TO THE SAME COMPOUND
    # on which to run the classifier
    # returns a proposed classifications
    def top_classification_aggregate(self,phrases):
        if len(phrases)==0:
            return 'Not Enough Data'
        agg_probs = self.confidence_classification_aggregate(phrases)
        return max(agg_probs, key=agg_probs.get)
    
    
        
        