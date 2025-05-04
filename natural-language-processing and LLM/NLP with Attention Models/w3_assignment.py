# In this assignment you will explore question answering. 
# You will implement the "Text to Text Transfer from Transformers" (better known as T5).

# Due to memory constraints of the jupyter environment and for the sake of time, your model will be trained with small datasets, so you won't get models that you could use in production but you will gain the necessary knowledge about how the Generative Language models are trained and used.
# You won't spend too much time with the architecture of the models but you will instead take a model that is pre-trained on a larger dataset and fine-tune it to get better results.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import traceback
import time
import json
from termcolor import colored
import string
import textwrap
import itertools
import numpy as np
import tensorflow_text as tf_text
import tensorflow as tf

import transformer_utils 
import utils

# Will come in handy later
wrapper = textwrap.TextWrapper(width=70)

# Set random seed
np.random.seed(42)

import w3_unittest

# In the initial phase of training a T5 model for a Question Answering task, the pre-training process involves leveraging a masked language model (MLM) on a very large dataset, such as the C4 dataset.
# The objective is to allow the model to learn contextualized representations of words and phrases, fostering a deeper understanding of language semantics.
# Before delving into pre-training, the dataset needs to be tokenized into smaller units, such as subwords or words, to facilitate model input.
# For the masked language modeling objective, a percentage of the tokenized input is randomly masked, and the model is trained to predict the original content of these masked tokens.

# Tokenizes text into subword units, enhancing handling of complex word structures, out-of-vocabulary words, and multilingual support.
with open("./models/sentencepiece.model", "rb") as f:
    pre_trained_tokenizer = f.read()
tokenizer = tf_text.SentencepieceTokenizer(pre_trained_tokenizer, out_type=tf.int32)
# The tokenizer does not add the EOS to the end of each sentence, so you need to add it manually
eos = tokenizer.string_to_id("</s>").numpy()

# Now you will create input and target pairs that will allow you to train your model.

# GRADED FUNCTION: tokenize_and_mask
def tokenize_and_mask(text, 
                      noise=0.15, 
                      randomizer=np.random.uniform, 
                      tokenizer=None):
    """Tokenizes and masks a given input.

    Args:
        text (str or bytes): Text input.
        noise (float, optional): Probability of masking a token. Defaults to 0.15.
        randomizer (function, optional): Function that generates random values. Defaults to np.random.uniform.
        tokenizer (function, optional): Tokenizer function. Defaults to tokenize.

    Returns:
        inps, targs: Lists of integers associated to inputs and targets.
    """
    
    # Current sentinel number (starts at 0)
    cur_sentinel_num = 0
    
    # Inputs and targets
    inps, targs = [], []

    # Vocab_size
    vocab_size = int(tokenizer.vocab_size())
    
    # EOS token id 
    # Must be at the end of each target!
    eos = tokenizer.string_to_id("</s>").numpy()
    
    ### START CODE HERE ###
    
    # prev_no_mask is True if the previous token was NOT masked, False otherwise
    # set prev_no_mask to True
    prev_no_mask = True
    
    # Loop over the tokenized text
    for token in tokenizer.tokenize(text).numpy():
        
        # Generate a random value between 0 and 1
        rnd_val = randomizer() 
        
        # Check if the noise is greater than a random value (weighted coin flip)
        if noise > rnd_val:
            
            # Check if previous token was NOT masked
            if prev_no_mask:
                
                # Current sentinel increases by 1
                cur_sentinel_num += 1
                
                # Compute end_id by subtracting current sentinel value out of the total vocabulary size
                end_id = vocab_size - cur_sentinel_num
                
                # Append end_id at the end of the targets
                targs.append(end_id)
                
                # Append end_id at the end of the inputs
                inps.append(end_id)
                
            # Append token at the end of the targets
            targs.append(token)
            
            # set prev_no_mask accordingly
            prev_no_mask = False

        else:
            
            # Append token at the end of the inputs
            inps.append(token)
            
            # Set prev_no_mask accordingly
            prev_no_mask = True
    
    
    # Add EOS token to the end of the targets
    targs.append(eos)
    
    ### END CODE HERE ###
    
    return inps, targs

inputs_targets_pairs = [tokenize_and_mask(text.encode('utf-8', errors='ignore').decode('utf-8'), tokenizer=tokenizer) 
                        for text in natural_language_texts[0:2000]]

def display_input_target_pairs(inputs_targets_pairs, sentinels, wrapper=textwrap.TextWrapper(width=70), tokenizer=tokenizer):
    for i, inp_tgt_pair in enumerate(inputs_targets_pairs, 1):
        inps, tgts = inp_tgt_pair
        inps = str(pretty_decode(inps, sentinels, tokenizer).numpy(), encoding='utf-8')
        tgts = str(pretty_decode(tgts, sentinels, tokenizer).numpy(), encoding='utf-8')
        print(f'[{i}]\n\n'
              f'inputs:\n{wrapper.fill(text=inps)}\n\n'
              f'targets:\n{wrapper.fill(text=tgts)}\n\n\n')

# Print 3 samples. We print inputs with less than 100 tokens. It is just to give you and idea of the process
display_input_target_pairs(filter(lambda x: len(x[0]) < 100, inputs_targets_pairs[0:12]), sentinels, wrapper, tokenizer)
"""Output:
[1]
inputs:
<Z>il plaid <Y>lycra <X> spandex shortall with metallic slinky
<W>sets. Attache <V> metallic elastic belt with O <U>ring. Head <T>
included. Great hip hop<S> jazz dance costume.<R> in the USA.
targets:
<Z> Fo <Y>  <X> and <W> in <V>d <U>- <T>band<S> or<R> Made

[2]
inputs:
I thought I was going to <Z> 3rd season <Y> Wire tonight. <X> there
was a commentary <W> 11, so I had to re <V>watch <U> Ground with <T>
commentary. Hopefully<S> can finish<R> season <Q>.
targets:
<Z> finish the <Y> of the <X> But <W> on episode <V>- <U> Middle <T>
the<S> I<R> the <Q> next weekend
"""



