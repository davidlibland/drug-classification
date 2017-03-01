#!/usr/bin/env python3

import time
import tensorflow as tf
import numpy as np
from .model import RNNClassifierArgs, RNNClassifierModel
import os
from . import config
from .translator import Translator
import dill



def setup_and_run(translator_filename):
    global translatorObj
    print("-"*30)
    print("Data Input")
    print("-"*30)
    # Load the translator object:
    translatorObj = Translator(filename = translator_filename)

    print("number of training examples: %d, vocab size: %d, number of classes: %d"
     %(translatorObj.data_len,len(translatorObj.vocab),len(translatorObj.target_to_id)))
     
    # Setup the args...
    args=RNNClassifierArgs(word_vocab_size = len(translatorObj.vocab),num_drug_classes = len(translatorObj.target_to_id),
            learning_rate=config.LEARNING_RATE,init_scale=1/(config.HIDDEN_SIZE), # Xavier Initialization
            num_steps=config.NUM_STEPS,num_layers=config.NUM_LAYERS,
            batch_size=config.BATCH_SIZE,keep_prob=config.KEEP_PROB,
            word_embedding_size=config.WORD_EMBEDDING_SIZE,
            hidden_size=config.HIDDEN_SIZE)
            
    # Now save the args so we can reload it when classifying
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    with open(os.path.join(args.log_dir,"args.pkl"),'wb') as f:
        dill.dump({'args':args},f)
    train_rnn(args)
    tf.reset_default_graph() 

def train_rnn(args):
    #if restore:
    #    load_rnn()
    print("-"*30)
    print("initialization")
    print("-"*30)
    
    # Initializer
    initializer = tf.random_uniform_initializer(-args.init_scale,args.init_scale)
    with tf.variable_scope("model", reuse = None, initializer = initializer):
        p = RNNClassifierModel(args=args, is_training=True, verbose=True)
                                            
    with tf.Session() as sess:
        print("-"*30)
        print("training")
        print("-"*30)
        
        # Initialize the variables
        tf.initialize_all_variables().run()
        
        saver = tf.train.Saver()
        for i in range(config.MAX_EPOCHS):
            epoch_size = (((translatorObj.data_len) // args.batch_size) - 1)# // args.num_steps
            print("epoch_size: ",epoch_size)
            #Shuffle the data:
            translatorObj.suffle_training_set()
            
            start_time = time.time()
            costs = 0.0
            iters = 0
            accuracy = 0.0
            # we backpropogate over a fixed number (num_steps) of GRU units, but we save the final_state
            # so that we can train the rnn to remember things over a much longer string.
            for step, (x, y) in enumerate(translatorObj.id_iterator(args.batch_size)):
                state = sess.run(p.initial_state)
                summary, cost_on_iter, cur_acc, state, _ = sess.run([p.merged, p.cost, p.accuracy, p.final_state, p.train_op],
                                         {p.input_IDs: x,
                                          p.target_ID: y,
                                          p.initial_state: state})
                costs += cost_on_iter
                if step !=0:
                    accuracy = ((step-1)*accuracy + cur_acc)/step
                else:
                    accuracy = cur_acc
                #iters += args.num_steps

                if step % (epoch_size // 10) == 10:
                    print("%.3f, accuracy: %.4f, emp. cross entr.: %.3f speed: %.0f samples/s, epoch est. time rem: %.0f" %
                        (step * 1.0 / epoch_size, accuracy, np.exp(-costs / step),
                         step * args.batch_size / (time.time() - start_time), (time.time() - start_time)*epoch_size/(step*1.0)))
                    p.writer.add_summary(summary, i)
            p.writer.flush()
            
            # Save the model every 10 epochs:
            if i>0  and i % 10==0:
                # Save the model in case we want to load it later...
                save_path = saver.save(sess, os.path.join(args.log_dir,"model.ckpt"),global_step=i)
                print("Model saved in file: %s" % save_path)
            
        # Save the model one final time.
        # Save the model in case we want to load it later...
        save_path = saver.save(sess, os.path.join(args.log_dir,"model.ckpt"),global_step=i)
        print("Model saved in file: %s" % save_path)