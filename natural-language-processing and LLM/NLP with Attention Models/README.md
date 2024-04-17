# NLP with Attention Models
## Table of contents
- [DeepLearning.AI: Natural Language Processing Specialization: NLP with Attention Models](#DeepLearningAI-Natural-Language-Processing-Specialization-NLP-with-Attention-Models)
  - [Week 1: Neural Machine Translation](#Week-1-Neural-Machine-Translation)
- [Resources](#Resources)

## DeepLearning.AI: Natural Language Processing Specialization: NLP with Attention Models
### Week 1: Neural Machine Translation
#### Seq2seq
The seq2seq model was traditionally used for the implementation of neural machine translation. It was created by Google in 2014.

In this model the inputs and outputs can have differing lengths. LSTMs and GRUs are used by seq2seq to avoid vanishing and exploding gradients. Before translation/decoding, encoding takes place, where sequences/sentences of variable length are transformed into a fixed length vector which encodes the overall meaning of the sentence. 

The seq2seq model uses an encoder and decoder to translate from one language to another. 

An encoder takes word tokens as input and returns its final hidden state as output. This hidden state is used by the decoder to generate the translated sentence.<br>
An encoder usually consists of an embedding layer, followed by a LSTM module consisting of one or more layers. The embedding layer transforms word tokens into numerical vectors for input to the LSTM module. The LSTM module outputs its final hidden state which encodes the overall meaning of the sentence.<br>
![Screenshot 2024-04-15 at 19 01 02](https://github.com/artainmo/machine-learning/assets/53705599/d99a7da9-907f-4d78-9c91-190eddf3973b)

The decoder is similarly constructed with an embedding layer and LSTM module. The output hidden state from encoder is used as previous hidden state of decoder's first step. Its input sequence starts with a `<sos>` (start of sequence) token to predict the first translated word. The predicted translated word is used as input for next step. 
![Screenshot 2024-04-15 at 19 02 04](https://github.com/artainmo/machine-learning/assets/53705599/459145c9-8d91-49d0-8b0c-096bf5496e1b)

One major limitation of the traditional seq2seq model is the 'information bottleneck'. Since seq2seq uses a fixed length memory for the hidden state, long sequences become problematic. Because no matter the input sequence length, only a fixed amount of information can be passed from the encoder to the decoder. Thus the model's performance diminishes as sequence size increases.<br>
One workaround is to use the hidden state of each word in the encoder instead of only using the final output hidden state. For this, an attention layer will be used to process those hidden states before passing them to the decoder.

#### Seq2seq model with attention
The attention layer specifies where to focus when translating. For example, when translating one paragraph from English to French, you can focus on translating one sentence at a time or even a couple of words at a time. The attention layer was initially developed for machine translation but is now used successfully in other domains too.

Seq2seq without attention starts to diminish in performance when reaching sentences longer than 20 words. This is because without attention, seq2seq needs to store the whole sequence's meaning in a fixed size vector. Seq2seq with attention performs as well no matter the input sequence length. This is because it is able to focus on specific inputs to predict words in the output translation, instead of having to memorize the entire input sentence into a fixed size vector.

Instead of passing only the final hidden state, you can pass all the hidden states to the decoder. However, this is expensive on memory. Instead you can combine all those hidden states into one vector, we call the 'context vector', by adding all the hidden state vectors up. Now the decoder is receiving information about each encoding step while it only needs information from the first few encoding steps to predict/decode the first word. The solution is to give more weight before the addition to encoding vectors representing words important for the next decoder step. This way the context vector is biased in containing more information about the most important words for next decoder step.<br>
The attention layer will calculate the weights and context vector.

First we calculate alignment indicated by e<sub>ij</sub>. It represents how well the inputs at j matches the expected output i. A higher score indicates a higher match. We do this using a feedforward neural network, taking the encoder and decoder hidden states as inputs. Its weights and biases are learned along with the rest of the seq2seq model. The scores are then turned into weights which range from zero to one using the softmax function. This means the weights can be thought of as a probability distribution which sum to one. Finally, each encoder hidden state is multiplied by its respective weights before summing all resulting vectors into one context vector.

## Resources
* [DeepLearning.AI - Natural Language Processing Specialization: Natural Language Processing with Attention Models](https://www.coursera.org/learn/attention-models-in-nlp)