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

import transformer_utils # Transformer from previous assignment
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

# Instead of training the question answering model from scratch, you will first "pre-train" the model using the C4 data set you just processed. 
# This will help the model to learn the general structure of language from a large dataset. 
# This is much easier to do, as you don't need to label any data, but just use the masking, which is done automatically.

# Define the model parameters
num_layers = 2
embedding_dim = 128
fully_connected_dim = 128
num_heads = 2
positional_encoding_length = 256

encoder_vocab_size = int(tokenizer.vocab_size())
decoder_vocab_size = encoder_vocab_size

# Initialize the model
transformer = transformer_utils.Transformer(
    num_layers, 
    embedding_dim, 
    num_heads, 
    fully_connected_dim,
    encoder_vocab_size, 
    decoder_vocab_size, 
    positional_encoding_length, 
    positional_encoding_length,
)

# define the optimizer and the loss function
learning_rate = transformer_utils.CustomSchedule(embedding_dim)
optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
train_loss = tf.keras.metrics.Mean(name='train_loss')
losses = [] # Here you will store the losses, so you can later plot them

# Limit the size of the input and output data so this can run in this environment
encoder_maxlen = 150
decoder_maxlen = 50

# you need to be sure that all inputs have the same length by truncating the longer sequences and padding the shorter ones with 0. The same must be done for the targets.
inputs = tf.keras.preprocessing.sequence.pad_sequences([x[0] for x in inputs_targets_pairs], maxlen=encoder_maxlen, padding='post', truncating='post')
targets = tf.keras.preprocessing.sequence.pad_sequences([x[1] for x in inputs_targets_pairs], maxlen=decoder_maxlen, padding='post', truncating='post')
inputs = tf.cast(inputs, dtype=tf.int32)
targets = tf.cast(targets, dtype=tf.int32)

# Create the final training dataset.
BUFFER_SIZE = 10000
BATCH_SIZE = 64
dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Now, you can run the training loop for 10 epochs. Running it with a big dataset such as C4 on a good computer with enough memory and a good GPU could take more than 24 hours. Here, you will run few epochs using a small portion of the C4 dataset for illustration. It will only take a few minutes, but the model won't be very powerful.

# Define the number of epochs
epochs = 10

# Training loop
for epoch in range(epochs):
    start = time.time()
    train_loss.reset_states()
    number_of_batches=len(list(enumerate(dataset)))
    for (batch, (inp, tar)) in enumerate(dataset):
        print(f'Epoch {epoch+1}, Batch {batch+1}/{number_of_batches}', end='\r')
        transformer_utils.train_step(inp, tar, transformer, loss_object, optimizer, train_loss)
    print (f'Epoch {epoch+1}, Loss {train_loss.result():.4f}')
    losses.append(train_loss.result())
    print (f'Time taken for one epoch: {time.time() - start} sec')

# Save the pretrained model
transformer.save_weights('./model_c4_temp')

# To show how powerful this model actually is, we trained it for several epochs with the full dataset in Colab and saved the weights for you. You can load them.
transformer.load_weights('./pretrained_models/model_c4')

# Now, you are going to fine tune the pretrained model for Question Answering using the SQUad 2.0 dataset.
# SQuAD, short for Stanford Question Answering Dataset, is a dataset designed for training and evaluating question answering systems. It consists of real questions posed by humans on a set of Wikipedia articles, where the answer to each question is a specific span of text within the corresponding article.

# GRADED FUNCTION: parse_squad
def parse_squad(dataset):
    """Extract all the answers/questions pairs from the SQuAD dataset

    Args:
        dataset (dict): The imported JSON dataset

    Returns:
        inputs, targets: Two lists containing the inputs and the targets for the QA model
    """

    inputs, targets = [], []

    ### START CODE HERE ###
    
    # Loop over all the articles
    for article in dataset:
        
        # Loop over each paragraph of each article
        for paragraph in article["paragraphs"]:
            
            # Extract context from the paragraph
            context = paragraph['context']
            
            #Loop over each question of the given paragraph
            for qa in paragraph['qas']:
                
                # If this question is not impossible and there is at least one answer
                if len(qa['answers']) > 0 and not(qa['is_impossible']):
                    
                    # Create the question/context sequence
                    question_context = 'question: ' + qa['question'] + ' context: ' + context
                    
                    # Create the answer sequence. Use the text field of the first answer
                    answer = 'answer: ' + qa['answers'][0]['text']
                    
                    # Add the question_context to the inputs list
                    inputs.append(question_context)
                    
                    # Add the answer to the targets list
                    targets.append(answer)
    
    ### END CODE HERE ###
    
    return inputs, targets

with open('data/train-v2.0.json', 'r') as f:
    example_jsons = json.load(f)
example_jsons = example_jsons['data']

inputs, targets =  parse_squad(example_jsons) 
# 40K pairs for training
inputs_train = inputs[0:40000] 
targets_train = targets[0:40000]  
# 5K pairs for testing
inputs_test = inputs[40000:45000] 
targets_test =  targets[40000:45000] 

# Limit the size of the input and output data so this can run in this environment
encoder_maxlen = 150
decoder_maxlen = 50
# You will first tokenize the inputs and the targets
inputs_str = [tokenizer.tokenize(s) for s in inputs_train]
targets_str = [tf.concat([tokenizer.tokenize(s), [1]], 0) for s in targets_train]
# You will ensure that the inputs and the outputs have the required lengths. Remember that the sequences longer than the required size will be truncated and the shorter ones will be padded with 0
inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs_str, maxlen=encoder_maxlen, padding='post', truncating='post')
targets = tf.keras.preprocessing.sequence.pad_sequences(targets_str, maxlen=decoder_maxlen, padding='post', truncating='post')
inputs = tf.cast(inputs, dtype=tf.int32)
targets = tf.cast(targets, dtype=tf.int32)
# Create the final training dataset.
BUFFER_SIZE = 10000
BATCH_SIZE = 64
dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# As usual, fine tuning this model to get state of the art results would require more time and resources than there are available in this environment
epochs = 2
losses = []

# Training loop
for epoch in range(epochs):
    start = time.time()
    train_loss.reset_states()
    number_of_batches=len(list(enumerate(dataset)))
    for (batch, (inp, tar)) in enumerate(dataset):
        print(f'Epoch {epoch+1}, Batch {batch+1}/{number_of_batches}', end='\r')
        transformer_utils.train_step(inp, tar, transformer, loss_object, optimizer, train_loss)
    print (f'Epoch {epoch+1}, Loss {train_loss.result():.4f}')
    losses.append(train_loss.result())
    print (f'Time taken for one epoch: {time.time() - start} sec')

# Save the final model
#transformer.save_weights('./pretrained_models/model_qa_temp')

# To get a model that works properly, you would need to train for about 100 epochs. So, we have pretrained a model for you. Just load the weights in the current model and let's use it for answering questions.
transformer.load_weights('./pretrained_models/model_qa3')

# GRADED FUNCTION: answer_question
def answer_question(question, model, tokenizer, encoder_maxlen=150, decoder_maxlen=50):
    """
    A function for question answering using the transformer model
    Arguments:
        question (tf.Tensor): Input data with question and context
        model (tf.keras.model): The transformer model
        tokenizer (function): The SentencePiece tokenizer
        encoder_maxlen (number): Max length of the encoded sequence
        decoder_maxlen (number): Max length of the decoded sequence
    Returns:
        _ (str): The answer to the question
    """
    
    ### START CODE HERE ###
    
    # QUESTION SETUP
    
    # Tokenize the question
    tokenized_question = tokenizer.tokenize(question)
    
    # Add an extra dimension to the tensor
    tokenized_question = tf.expand_dims(tokenized_question, 0) 
    
    # Pad the question tensor
    padded_question = tf.keras.preprocessing.sequence.pad_sequences(tokenized_question,
                                                                    maxlen=encoder_maxlen,
                                                                    padding='post', 
                                                                    truncating='post') 
    # ANSWER SETUP
    
    # Tokenize the answer
    # Hint: All answers begin with the string "answer: "
    tokenized_answer = tokenizer.tokenize("answer: ")
    
    # Add an extra dimension to the tensor
    tokenized_answer = tf.expand_dims(tokenized_answer, 0)
    
    # Get the id of the EOS token
    eos = tokenizer.string_to_id("</s>") 
    
    # Loop for decoder_maxlen iterations
    for i in range(decoder_maxlen):
        
        # Predict the next word using the model, the input document and the current state of output
        next_word = transformer_utils.next_word(padded_question, tokenized_answer, model)
        
        # Concat the predicted next word to the output 
        tokenized_answer = tf.concat([tokenized_answer, next_word], axis=1)
        
        # The text generation stops if the model predicts the EOS token
        if next_word == eos:
            break 
    
    ### END CODE HERE ###

    return tokenized_answer 

idx = 110
result = answer_question(inputs_test[idx], transformer, tokenizer)


