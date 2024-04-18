# NLP with Attention Models
## Table of contents
- [DeepLearning.AI: Natural Language Processing Specialization: NLP with Attention Models](#DeepLearningAI-Natural-Language-Processing-Specialization-NLP-with-Attention-Models)
  - [Week 1: Neural Machine Translation](#Week-1-Neural-Machine-Translation)
- [Resources](#Resources)

## DeepLearning.AI: Natural Language Processing Specialization: NLP with Attention Models
### Week 1: Neural Machine Translation
#### Seq2seq
The seq2seq model was traditionally used for the implementation of neural machine translation (NMT). It was created by Google in 2014.

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

First we calculate alignment indicated by e<sub>ij</sub>. It represents how well each encoder hidden state at j matches the previous decoder hidden state at (i - 1). A higher score indicates a higher match. We do this using a feedforward neural network, taking the encoder hidden states and decoder hidden state as inputs. Its weights and biases are learned along with the rest of the seq2seq model. The scores are then turned into weights which range from zero to one using the softmax function. This means the weights can be thought of as a probability distribution which sum to one. Finally, each encoder hidden state is multiplied by its respective weights before summing all resulting vectors into one context vector.

#### Queries, Keys, Values and Attention
New variations exist of the attention layer. In 2017 a paper introduced an efficient form of attention based on information retrieval, using queries, keys and values. It is called the scaled dot product attention or QKV (query, keys, values) attention. 

Conceptually, you can think of keys and values as a lookup table. A query is matched to a key and the associated value is returned. For example when translating the French word 'heure' it matches with the English key 'time' and its value being a numerical vector is returned.<br>
In practice however, queries, keys and values are all represented by numerical vectors. This allows the model to learn which words are most similar between source and target language. The similarity between words we call alignment. The query and key vectors are used to calculate alignment scores that measures how well the query and keys match. In the end, the alignment scores are turned into weights who are used for summing the value vector. The weighted sum of the value vector becomes the attention vector.<br>
This process can be performed using scaled dot-product attention. The queries of each step are packed together into a matrix so that the attention can be computed simultaneously for each query. The keys and values are also packed into their own matrices and used as inputs together with the query matrix for the attention function.<br>
The attention function goes as follows. First the query and key matrices are multiplied to get a matrix of alignment scores. These are then divided by the square root of the key vector size for regularization. The result of this we transform into weights using the softmax function. Finally, the weight and value matrices are multiplied to get the attention vectors for each query.

While in the previously seen original form of attention we used a feedforward neural network, in the scaled dot product attention seen here we only use two matrix multiplications. Since matrix multiplication is highly optimized in modern deep learning frameworks, this form of attention is much faster to compute.

In the scaled dot product attention, usually the alignments between source and target languages are not learned in attention layer but instead in the embedding layer or other layer that comes before the attention layer. The alignment weights form a matrix where queries lie on the rows and keys on the columns. Each entry in this matrix is the weight for the corresponding query-key pair. Similar words will have larger weights. Via training, the model learns which words have similar meanings and encodes that information in the query and key vectors. Learning alignment like this is beneficial for translating between languages with different grammatical structures. Since attention looks at the entire input and target sentences at once and calculates alignments based on word pairs, weights are assigned appropriately regardless of word order. For example in the following picture you can see two phrases where words with similar meanings 'zone' and 'area' don't occur in same word order in phrase but using a matrix we can still detect their similarity/alignment.<br>
![Screenshot 2024-04-18 at 13 47 50](https://github.com/artainmo/machine-learning/assets/53705599/82e60118-f8a5-4e95-8d73-08501e4a8812)

#### Machine Translation Setup
Start by using pre-trained vector embeddings. Or else represent words initially with [one-hot-vectors](https://github.com/artainmo/machine-learning/tree/main/natural-language-processing%20and%20LLM/NLP%20with%20Probabilistic%20Models#word-representations).<br>
Usually you'll keep track of index mappings with index to word (ind2word) and word to index (word2ind) dictionaries.<br>
Also you will normally use an end of sequence token/symbol \<EOS\>.<br>
Token vectors need to be padded wih zeros to match the length of the longest sequence.

#### Teacher forcing
Teacher forcing is a concept used in training a neural machine translation (NMT) model to improve speed of training and accuracy.

Seq2seq models generate translation by feeding the output of a decoder step as input of the next decoder step.<br>
Intuitively, you would calculate the loss by comparing the decoder output sequence with the target sequence. Calculate the cross entropy loss for each step and sum those for the total loss.<br>
However this does not work well in practice because in early training the model makes a lot of wrong predictions and those wrong predictions are used as input of next decoding steps, thus not giving a chance for the next predictions to be right, creating excessively large losses for steps later in the sequence.<br>
The solution is to use the target sequence words as inputs of the decoding steps. Giving them the correct inputs allows them to have a chance in producing correct predictions. This speeds up training by a lot and is what we call 'teacher forcing'.

There are some variations of this. For example, later in training when predictions start to have more accuracy, decoder outputs can be used again instead of target words. This is known as 'curriculum learning'.

#### Train NMT Model with Attention
Previously we have seen a model that looks like this.<br>
![Screenshot 2024-04-18 at 18 45 01](https://github.com/artainmo/machine-learning/assets/53705599/99491e23-9bc7-452e-8986-7b0673fbc905)<br>
Recall that the decoder is supposed to pass its hidden states to the attention mechanism for the attention mechanism to produce context vectors who are used by the decoder to produce hidden states. To resolve this issue we will use two decoders instead. A pre-attention decoder to provide hidden states and a post-attention decoder for the translations.

The new model containing two decoders looks like this.<br>
![Screenshot 2024-04-18 at 18 54 20](https://github.com/artainmo/machine-learning/assets/53705599/2a2abde7-cb83-4d8b-9754-92e352b5314e)

Masking is a way to tell sequence-processing layers that certain timesteps in an input are missing, and thus should be skipped when processing the data.<br>
Padding is a special form of masking where the masked steps are at the start or the end of a sequence.

Now, let's take a closer look at each piece of the model. The initial step is to make two copies of the input tokens and the target tokens because you will need them in different places of the model.<br>
Within the pre-attention decoder, you shift each sequence to the right and add a start of sentence token.<br>
In the encoder and pre-attention decoder, the inputs and targets go through an embedding layer before going through LSTMs.<br>
After getting the query, key and value vectors from the encoder and pre-attention decoder, you have to prepare them for the attention layer. You'll use a function to help you get a padding mask to help the attention layer determine the padding tokens. This step is where you will use the copy of the input tokens.<br>
Now, everything is ready for attention. You pass the queries, keys, values, and the mask to the attention layer that outputs the context vector and the mask.<br>
Before going through the decoder, you drop the mask. You then pass the context vectors through the decoder composed of an LSTM, a [dense layer](https://github.com/artainmo/machine-learning/tree/main/natural-language-processing%20and%20LLM/NLP%20with%20Sequence%20Models#different-layers), and a [LogSoftmax](https://github.com/artainmo/machine-learning/tree/main/supervised-learning%20and%20neural-networks#activation-function).<br>
In the end, your model returns log probabilities and the copy of the target tokens that you made at the beginning.<br>
![Screenshot 2024-04-18 at 19 08 41](https://github.com/artainmo/machine-learning/assets/53705599/b8db6fbc-f2f6-4f1a-8eb9-b6162362de21)

#### BLEU Score
Bilingual evaluation understudy (BLEU) is one metric among others used to evaluate machine translation models. It evaluates the quality of machine-translated candidate text by comparing it to one or more references who usually are human translations. The closer to 1 the BLEU score, the better the model is, while the closer to 0, the worse it is.

To get the BLUE score you have to compute the precision of the candidate translation by comparing its [n-grams](https://github.com/artainmo/machine-learning/tree/main/natural-language-processing%20and%20LLM/NLP%20with%20Probabilistic%20Models#n-grams) with reference translations.<br>
Here is a demonstration using unigrams. You basically need to count the number of unigrams from the candidate translation that appear in any of the references and divide that count by the total amount of words in the candidate translation.<br>
![Screenshot 2024-04-18 at 23 45 18](https://github.com/artainmo/machine-learning/assets/53705599/bb363e6d-dcd0-40b0-b75e-7f72ced1ec7d)<br>
In this example a bad translation got a perfect score. This is because the bad translation consisted of common words only. A modified BLEU score could prevent this erroneous score.

For the modified version of the BLEU score, after you find a word from the candidates in one or more of the references, you stop considering that word from the reference for the following words in the candidates. In other words, you exhaust the words in the references after you match them with a word in the candidates.
![Screenshot 2024-04-18 at 23 58 41](https://github.com/artainmo/machine-learning/assets/53705599/3f50b72c-d125-4d8b-95aa-5a9cec3c990c)

BLEU score is the most widely adopted evaluation metric for machine translation. However you should be aware of its limitations. It does not consider semantic meaning, neither sentence structure.

## Resources
* [DeepLearning.AI - Natural Language Processing Specialization: Natural Language Processing with Attention Models](https://www.coursera.org/learn/attention-models-in-nlp)
* [Understanding masking & padding](https://www.tensorflow.org/guide/keras/understanding_masking_and_padding#:~:text=Masking%20is%20a%20way%20to,the%20end%20of%20a%20sequence.)
