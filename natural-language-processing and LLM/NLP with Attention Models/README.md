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

The decoder is similarly constructed with an embedding layer and LSTM module. The output hidden state from encoder is used as previous hidden state of decoder's first step. Its input sequence starts with a `<sos>` (start of sequence) token to predict the first translated word. The predicted translated word is used as input for next step.<br>
![Screenshot 2024-04-15 at 19 02 04](https://github.com/artainmo/machine-learning/assets/53705599/459145c9-8d91-49d0-8b0c-096bf5496e1b)

One major limitation of the traditional seq2seq model is the 'information bottleneck'. Since seq2seq uses a fixed length memory for the hidden state, long sequences become problematic. Because no matter the input sequence length, only a fixed amount of information can be passed from the encoder to the decoder. Thus the model's performance diminishes as sequence size increases.<br>
One workaround is to use the hidden state of each word in the encoder instead of only using the final output hidden state. For this, an attention layer will be used to process those hidden states before passing them to the decoder.

#### Seq2seq model with attention
The attention layer specifies where to focus when translating. The attention layer was initially developed for machine translation but is now used successfully in other domains too.

Seq2seq without attention starts to diminish in performance when reaching sentences longer than 20 words. This is because without attention, seq2seq needs to store the whole sequence's meaning in a fixed size vector. Seq2seq with attention performs as well no matter the input sequence length. This is because it is able to focus on specific inputs to predict words in the output translation, instead of having to memorize the entire input sentence into a fixed size vector.

Instead of passing only the final hidden state, you can pass all the hidden states to the decoder. However, this is expensive on memory. Instead you can combine all those hidden states into one vector, we call the 'context vector', by adding all the hidden state vectors up. Now the decoder is receiving information about each encoding step while it only needs information from the first few encoding steps to predict/decode the first word. The solution is to give more weight before the addition to encoding vectors representing words important for the next decoder step. This way the context vector is biased in containing more information about the most important words for next decoder step.<br>
The attention layer will calculate the weights and context vector.

First we calculate alignment indicated by e<sub>ij</sub>. It represents how well each encoder hidden state at j matches the previous decoder hidden state at (i - 1). A higher score indicates a higher match. We do this using a feedforward neural network, taking the encoder hidden states and decoder hidden state as inputs. Its weights and biases are learned along with the rest of the seq2seq model. The scores are then turned into weights which range from zero to one using the softmax function. This means the weights can be thought of as a probability distribution which sum to one. Finally, each encoder hidden state is multiplied by its respective weights before summing all resulting vectors into one context vector.

#### Queries, Keys, Values and Attention
New variations exist of the attention layer. In 2017 a paper introduced an efficient form of attention based on information retrieval, using queries, keys and values. It is called the scaled dot product attention or QKV (query, keys, values) attention. 

Conceptually, you can think of keys and values as a lookup table. A query is matched to a key and the associated value is returned. For example when translating the French word 'heure' it matches with the English key 'time' and its value being a numerical vector is returned.<br>
In practice however, queries, keys and values are all represented by numerical vectors. This allows the model to learn which words are most similar between source and target language. The similarity between words we call alignment. The query and key vectors are used to calculate alignment scores that measures how well the query and keys match. In the end, the alignment scores are turned into weights who are used on the value vector to form the context vector.<br>
This process can be performed using scaled dot-product attention. The queries of each step are packed together into a matrix so that the attention can be computed simultaneously for each query. The keys and values are also packed into their own matrices and used as inputs together with the query matrix for the attention function.<br>
The attention function goes as follows. First the query and key matrices are multiplied to get a matrix of alignment scores. These are then divided by the square root of the key vector size for regularization. The result of this we transform into weights using the softmax function. Finally, the weight and value matrices are multiplied to get the context vectors for each query.

While in the previously seen original form of attention we used a feedforward neural network, in the scaled dot product attention seen here we only use two matrix multiplications. Since matrix multiplication is highly optimized in modern deep learning frameworks, this form of attention is much faster to compute.

In the scaled dot product attention, usually the alignments between source and target languages are not learned in attention layer but instead in the embedding layer or other layer that comes before the attention layer. The alignment weights form a matrix where queries lie on the rows and keys on the columns. Each entry in this matrix is the weight for the corresponding query-key pair. Similar words will have larger weights. Via training, the model learns which words have similar meanings and encodes that information in the query and key vectors. Learning alignment like this is beneficial for translating between languages with different grammatical structures. Since attention looks at the entire input and target sentences at once and calculates alignments based on word pairs where weights are assigned appropriately regardless of word order. For example in the following picture you can see two phrases where words with similar meanings 'zone' and 'area' don't occur in same word order in phrase but using a matrix we can still detect their similarity/alignment.<br>
![Screenshot 2024-04-18 at 13 47 50](https://github.com/artainmo/machine-learning/assets/53705599/82e60118-f8a5-4e95-8d73-08501e4a8812)

#### Machine Translation Setup
Start by using pre-trained vector embeddings. Or else represent words initially with [one-hot-vectors](https://github.com/artainmo/machine-learning/tree/main/natural-language-processing%20and%20LLM/NLP%20with%20Probabilistic%20Models#word-representations).<br>
Usually you'll keep track of index mappings with index to word (ind2word) and word to index (word2ind) dictionaries.<br>
Also you will normally use an end of sequence token/symbol \<EOS\>.<br>
Token vectors need to be padded wih zeros to match the length of the longest sequence.

#### Teacher forcing
Teacher forcing is a concept used in training a neural machine translation (NMT) model to improve speed of training and accuracy.

Seq2seq models generate translations by feeding the output of a decoder step as input of the next decoder step.<br>
Intuitively, you would calculate the loss by comparing the decoder output sequence with the target sequence. Calculate the cross entropy loss for each step and sum those for the total loss.<br>
However, this does not work well in practice because in early training the model makes a lot of wrong predictions and those wrong predictions are used as input of next decoding steps, thus not giving a chance for the next predictions to be right, creating excessively large losses for steps later in the sequence.<br>
The solution is to use the target sequence words as inputs of the decoding steps. Giving them the correct inputs allows them to have a chance in producing correct predictions. This speeds up training by a lot and is what we call 'teacher forcing'.

There are some variations of this. For example, later in training when predictions start to have more accuracy, decoder outputs can be used again instead of target words. This is known as 'curriculum learning'.

#### Train NMT Model with Attention
Previously we have seen a model that looks like this.<br>
![Screenshot 2024-04-18 at 18 45 01](https://github.com/artainmo/machine-learning/assets/53705599/99491e23-9bc7-452e-8986-7b0673fbc905)<br>
Recall that the decoder is supposed to pass its hidden states to the attention mechanism for the attention mechanism to produce context vectors who are used by the decoder to produce hidden states. To resolve this issue we will use two decoders instead. A pre-attention decoder to provide hidden states and a post-attention decoder for the translations.

The new model containing two decoders looks like this.<br>
![Screenshot 2024-04-18 at 18 54 20](https://github.com/artainmo/machine-learning/assets/53705599/2a2abde7-cb83-4d8b-9754-92e352b5314e)<br>

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

For the modified version of the BLEU score, after you find a word from the candidates in one or more of the references, you stop considering that word from the reference for the following words in the candidates. In other words, you exhaust the words in the references after you match them with a word in the candidates.<br>
![Screenshot 2024-04-18 at 23 58 41](https://github.com/artainmo/machine-learning/assets/53705599/3f50b72c-d125-4d8b-95aa-5a9cec3c990c)<br>

The table below shows the typical values of BLEU score multiplied by 100. 
| Score | Interpretation |
| -- | -- |
| < 10	| Almost useless |
| 10 - 19	| Hard to get the gist |
| 20 - 29	| The gist is clear, but has significant grammatical errors |
| 30 - 40	| Understandable to good translations |
| 40 - 50	| High quality translations |
| 50 - 60	| Very high quality, adequate, and fluent translations |
| > 60 | Quality often better than human |

<br>
BLEU score is the most widely adopted evaluation metric for machine translation. However you should be aware of its limitations. It does not consider semantic meaning, neither sentence structure.

#### ROUGE-N Score
The recall-oriented-understudy-of-gisting-evaluation (ROUGE) score is another performance metric estimating the quality of machine-translation. ROUGE was initially developed to evaluate the quality of machine summarized texts, but is also helpful in assessing the quality of machine translation. While BLEU measures how much of the candidate translation appears in the references, ROUGE measures how much of the references appears in the candidate translation. 

Different versions exist of ROUGE. Let's look at ROUGE-N. First, count how many n-grams from the references also appear in the candidate translation and divide this number by the number of n-grams in the reference. If having multiple references, get the ROUGE-N score for each reference and keep the maximum score which is 0.4 in following example.<br>
![Screenshot 2024-04-19 at 11 56 49](https://github.com/artainmo/machine-learning/assets/53705599/1c28544f-d073-4598-8f60-51deb7d4c022)

This basic version of the ROUGE-N score is based on [recall](https://github.com/artainmo/machine-learning/tree/main/supervised-learning%20and%20neural-networks#evaluation) while the BLEU score you saw previously is [precision](https://github.com/artainmo/machine-learning/tree/main/supervised-learning%20and%20neural-networks#evaluation) based. But why not combine both to get a metric like an [F1 score](https://github.com/artainmo/machine-learning/tree/main/supervised-learning%20and%20neural-networks#evaluation).<br>
![Screenshot 2024-04-19 at 12 07 25](https://github.com/artainmo/machine-learning/assets/53705599/65149d0d-f662-4eaa-b207-c264477e5afa)<br>
You have now seen how to compute the modified BLEU and the sample ROUGE-N scores to evaluate your model. You can view these metrics like precision and recall. Therefore, you can use both to get an F1 score that could better assess the performance of your machine translation model.

While the BLEU score does not consider semantic meaning, nor sentence structure, the ROUGE score neither.

#### Decoding and Sampling
Greedy decoding and random sampling are methods used to construct a sentence.

In a seq2seq model the decoder outputs at each step a probability distribution over all the words and symbols in the target vocabulary. The final output of the seq2seq model depends on what word you choose out of that probability distribution at each step.

Greedy decoding is the simplest way to select a word out of the model's predicted probability distribution as it selects the most probable word at every step. The problem with this method is that when handling long sequences it can select common words repeatedly and form a phrase like this "I am am am I".<br>

If for example in the probability distribution the word 'I' has a probability of 0.2 and word 'am' of 0.04, then with random sampling the word 'I' will be selected 20% of the time and 'am' 4% of the time. Random sampling basically randomly selects one of the words, knowing words with higher probabilities will also have a higher chance of being selected. The problem is that it is too random. Temperature is a parameter you can adjust to modulate the randomness when using random sampling. It is measured on a scale of 0-1, indicating low to high randomness.

#### Beam Search
Since taking the output with the highest probability at each time step is not ideal. We will look at beam search instead.

If you had infinite computational power, you could calculate the probability of every possible output sentence and choose the best one. In the real world we use beam search. This method attempts to find the most likely output sentence by choosing some number of best sequences based on conditional probabilities at each time step.

Now at each time step with beam search you have to calculate the probability of potential sequences given the outputs of the previous time step.<br>
Beam width (parameter B) is used to limit the possible sequences you will compute the probability of. At each step you only keep the B most probable sequences and drop all others. You keep generating new words until all the B most probable sequences end with the end of sequence \<EOS\> token/symbol.<br>
Technically, greedy decoding is the same as beam search with B equal to 1.

Here is how beam search can look like with B equal to 2. You start with a start of sequence \<sos\> token/symbol. Then you take the following two words with highest probability. For each of those words you branch out and evaluate their next words' probability distribution. Multiplying first and second word probabilities will give sequence probabilities. Choose the two sequences with highest probabilities. And so forth.<br>
![Screenshot 2024-04-19 at 13 24 56](https://github.com/artainmo/machine-learning/assets/53705599/8035bb5a-8640-4c02-9ec8-579c5718dd06)<br>

The vanilla version of beam search has some disadvantages.<br>
For instance it penalizes long sequences because the probability of a sequence is computed as the product of multiple conditional probabilities. A solution is to normalize the probability of each sequence by its sequence length.<br>
Depending on B size, beam search can be expensive computationally and on memory.

#### Minimum Bayes Risk
Minimum bayes risk (MBR) is a method that compares multiple candidate translations and selects the best one.

Begin by generating several candidate translations via random sampling with temperature for example, then compare each candidate translation against each other using a similarity score or a loss function. ROUGE would be a good choice. Finally, choose the sample with the highest average similarity or the lowest loss.<br>
![Screenshot 2024-04-19 at 15 31 08](https://github.com/artainmo/machine-learning/assets/53705599/e930af3f-46a0-4403-b9c1-6b92b966fa67)<br>
Here are the steps for implementing MBR with ROUGE on a small set of four candidate translations.<br>
![Screenshot 2024-04-19 at 15 42 27](https://github.com/artainmo/machine-learning/assets/53705599/aa958e42-681a-4451-88bc-5d8550a5d4e7)<br>
Finally, you select the candidate with the highest average ROUGE score and that's it for MBR.

MBR provides a more contextually accurate translation compared to random sampling and greedy decoding.

### Week 2: Text Summarization
This week we will use the transformer network for summarization. Summarization is an important task in NLP and it's useful for consumer enterprise. For example, bots can be used to scrape articles and summarize them. Then you can use sentiment analysis to identify the sentiment about certain topics in the articles.

We will also look at different types of attention, like dot-product attention, causal attention, encoder-decoder attention, and self-attention.

#### Transformers vs RNNs
The transformer model was created by Google in 2017 as a purely attention-based model to remediate some problems with RNNs.

In neural machine translation with RNNs the start of the text needs to be translated before the end. This means translation is done sequentially which leaves no room for parallel computing for speed.<br>
Besides speed, RNNs can also lose information over long sequences. For example it may not remember if the subject is singular or plural as it moves further away from the subject. Vanishing gradients can also occur on long sequences. However, those two last issues can already be mitigated with GRUs and LSTMs, but transformers do it even better.

Previously we saw encoders and decoders in neural machine translation made out of RNNs. In a transformer, encoders and decoders are not necessary as attention only is needed. As a result, computation does not need to be sequential and vanishing gradients don't occur on long sequences.

#### Transformers overview
The transformers revolutionized the field of natural language processing. The first transformer paper, named 'Attention is all you need', sets the basis for all the models we will view in this course. The transformer architecture has become the standard for large language models, including BERT, T5, and GPT-3.

The transformer model uses 'scaled dot-product attention' which we saw in the first week of this course. This form of attention is efficient in terms of computation and memory due to it consisting of just matrix multiplication operations, allowing transformers to grow larger while being faster and requiring less memory.

In the transformer model we will use the 'multi-head attention' layer. This layer runs in parallel to perform scaled dot-product attention and multiple linear transformations of the input queries, keys, and values. In this layer, the linear transformations are learnable parameters.<br>
![Screenshot 2024-04-22 at 20 09 47](https://github.com/artainmo/machine-learning/assets/53705599/7be3b015-4470-43e4-84fc-ffa4d8d3986d)<br>

The transformer encoder starts with a multi-head attention module to perform self-attention on the input sequence. Self-attention consists of every input item attending every other input item. This is followed by normalization, a feed forward layer and normalization again, to provide a contextual representation of each input. This entire block is one encoder layer and is repeated N number of times.<br>
![Screenshot 2024-04-22 at 20 40 17](https://github.com/artainmo/machine-learning/assets/53705599/6d4ec052-0289-4e80-b3f4-964eab808eba)<br>

The decoder is constructed with multi-head attention modules, normalization and a feed forward layer. The first attention module is masked such that each position attends only to previous positions. It blocks leftward flowing information. The second attention module takes the encoder output and allows the decoder to attend to all items. This whole decoder layer is also repeated some number of times, one after another.<br>
![Screenshot 2024-04-22 at 20 46 00](https://github.com/artainmo/machine-learning/assets/53705599/6f2a0687-feb7-4c7a-a4a2-af5736a7d23d)<br>

Transformers also incorporate a positional encoding stage which encodes each input's position in the sequence. This is necessary because transformers don't use recurrent neural networks, while the word order is still relevant for any language. Positional encoding can be learned or fixed, just as with word embeddings.<br>
The final model looks like this.<br>
![Screenshot 2024-04-22 at 20 53 23](https://github.com/artainmo/machine-learning/assets/53705599/7bbfdd9f-b719-45b7-a1c7-12a038819ef5)<br>
We will look more in depth at each part later in this course.

This architecture is easy to parallelize compared to RNN models, and as such, can be trained much more efficiently on multiple GPUs.

#### Transformer Applications
Since transformers can be applied to any sequential task just like RNNs, it has been widely used throughout NLP. It is often used for text summarization. But it can also be used for autocompletion, named entity recognition (NER), question answering, chat-bots, machine translation and many other NLP tasks.

GPT-2 stands for generative pre-training for transformer. It is a decoder-only transformer model created by OpenAI in 2018. It is the precursor of the now popular [chatGPT product](https://chatgpt.com/).<br>
BERT is a transformer model created by Google in 2018 used for learning text representations, it contains only an encoder.<br>
T5 is a multitask transformer model created by Google in 2019. Usually separate models would need separate training for separate tasks. However, the T5 transformer can perform multiple tasks such as question answering, translation and classification through one trained transformer model, by describing in input what task you want performed on what data. [Transfer learning](https://github.com/artainmo/DevOps/tree/main/cloud#generative-ai-transformers-gpt-self-attention-and-foundation-models) is the concept behind this. It can also perform regression and summarization. A regression model outputs a continuous numerical value. Here, regression can be used for example to provide a similarity score between two input sentences.

#### Scaled and Dot-Product Attention
The main operation in transformers is the [scaled dot-product attention we overviewed previously](#Queries-Keys-Values-and-Attention).

Recall that in scale dot-product attention, you have queries, keys and values. To get the query, key and value matrices, you must first transform the words in your sequences into embeddings. The query matrix will contain all the input word embeddings as rows while the key matrix will contain all the target word embeddings as rows. You will generally use the same vectors used for the key matrix for the value matrix.<br>
The attention layer outputs context vectors for each query using the following formula and taking as input the query (Q), the key (K), the key size (d<sub>k</sub>) and the value (V).<br>
![Screenshot 2024-04-22 at 22 53 18](https://github.com/artainmo/machine-learning/assets/53705599/a0b0f142-a3e7-4773-81b7-d0fe51595130)<br>

In the transformer decoder, we need an extended version of the scaled dot-product attention, called the self-masked attention.

#### Masked self-attention
Before reviewing self-attention we will first review the other attention mechanisms found in the transformer model.

In encoder-decoder attention all words in one sentence attend all words in another sentence. The queries come from one sentence while the keys and values come from another. We already saw this previously where for example, as in the image below, the English words attend the French words.<br>
![Screenshot 2024-04-23 at 11 14 32](https://github.com/artainmo/machine-learning/assets/53705599/ea8a41cb-2cb4-445f-81db-d060f441d31c)<br>

In self-attention, the queries, keys and values come from the same sentence. Thus every word attends to every other word in the sequence. This provides contextual word representations. Representations of the meaning of each word within the sentence.<br>
![Screenshot 2024-04-23 at 11 36 53](https://github.com/artainmo/machine-learning/assets/53705599/cdf6a175-4772-41a9-a3c1-a40a683f12bb)<br>

In masked self-attention, the queries, keys and values come from the same sentence. But queries cannot attend keys on future positions.<br>
![Screenshot 2024-04-23 at 11 37 26](https://github.com/artainmo/machine-learning/assets/53705599/2f5c54af-c1eb-4322-a2a7-e67d6c4057e2)<br>
This attention mechanism is present in the decoder from the transformer model and ensures that predictions at each position depend only on the known outputs.

Recall that the scale dot-product attention requires the calculation of the softmax of the scaled products between the queries and the transpose of the key matrix. Then for masked self-attention, you add a mask matrix within the softmax. The mask has a zero on all of its positions, except for the elements above the diagonal, which are set to minus infinity. Or in practice, a huge negative number. This addition ensures that weights assigned to future positions are equal to zero so that the queries don't attend future key positions. <br>
![Screenshot 2024-04-23 at 11 48 58](https://github.com/artainmo/machine-learning/assets/53705599/cf17d716-e21d-4288-beca-3162d52b8120)<br>
In the end, as with the other types of attention, you multiply the weights matrix by the value matrix to get the context vector for each query.

#### Multi-head attention
Multi-head attention allows parallel computation and thus more computations in the same amount of time compared to single-head attention.

In multi-head attention, the number of times that you apply the attention mechanism in parallel is the number of heads in the model. For example a model with two heads would need two sets of queries, keys and values. Recall that word embeddings are necessary to get the query, key and value matrices.<br>
You can get different sets/representations by linearly transforming the original embeddings, using a set of matrices W<sup>Q</sup>, W<sup>K</sup>, W<sup>V</sup>, for each head in the model. Using different sets/representations, allows your model to learn multiple relationships between the words from the query and key matrices.

First, queries, keys, and values are transformed into different sets/representations for the different heads. After performing the scaled dot-product attention on each set, you will concatenate the resulting sets into a single matrix. Finally, you transform that matrix to get the output context vectors.<br>
![Screenshot 2024-04-23 at 12 30 28](https://github.com/artainmo/machine-learning/assets/53705599/fbb1df9c-56ff-431d-93f7-9c08e1fd3920)<br>
Note that every linear transformation in multi-head attention contains a set of learnable parameters.

The inputs to the multi-head attention layer are the query, key, and value matrices. The number of columns in those matrices equals the embedding size, and the number of rows equals the input sequence length.<br>
Using sets of transformation matrices W<sup>Q</sup>, W<sup>K</sup>, W<sup>V</sup> we will transform the query, key and value matrices for each head. The number of rows in those transformation matrices equals the embedding size while the number of columns can be choosen. However, it is advised to use a number of columns equal to the embedding size divided by the number of heads, which ensures a computational cost of multi-head attention that doesn't exceed by much the one for single-head attention.<br>
After getting the transformed query, key and value matrices for each head, you can execute in parallel the attention mechanism. As a result, you get a matrix per head with the column dimensions equalling embedding size divided by heads, and the number of rows in those matrices being the same as the number of rows in the query matrix. Then you concatenate horizontally the matrices outputted by each attention head in the model. 
Lastly, you apply a linear transformation using W<sup>O</sup> on the concatenated matrix. W<sup>O</sup> has columns and rows equal to embedding size. The result is a matrix with the context vectors of column size equalling embedding size and rows equalling number of query rows.<br>
![Screenshot 2024-04-23 at 13 13 09](https://github.com/artainmo/machine-learning/assets/53705599/b82c056e-f384-4258-83da-be11ca678b48)<br>

#### Transformer Decoder
A transformer decoder takes a tokenized sentence as input. Those are transformed into word embeddings. We addition the positional encodings to those word embeddings to also provide word order information.<br>
This constitutes the input for the first multi-headed attention layer. After this attention layer we have a feedforward layer. After each attention and feedforward layer we need to addition the layer input with the layer output and normalize the result of this addition. The attention and feedforward layers are repeated N times. The original model started with N=6, but now transformers go up to 100 or even more.<br>
Lastly, we have a dense and softmax layer for output.<br>
![Screenshot 2024-04-23 at 17 46 28](https://github.com/artainmo/machine-learning/assets/53705599/d6bbb7b3-1bd1-4a9d-976a-744350f1f416)<br>
The attention mechanism searches relationships between words in the sequence and provides weights to those word relationships. The feedforward layer performs non-linear transformations and uses ReLu activation functions for each input. The feedforward neural network output vectors will essentially replace the hidden states of the original RNN decoder.

#### Transformer for Summarization
As input, our transformer model gets whole news articles for example. As output, our model is expected to produce a summary of those articles, that is, few sentences that mentions the most important ideas.

During training, the input for the model is a long text that starts with for example a news article, is followed by the EOS tag, the summary of that article, and another EOS tag. The summary is added to the input during supervised training to form a labeled dataset.<br>
![Screenshot 2024-04-23 at 18 37 59](https://github.com/artainmo/machine-learning/assets/53705599/28d9b284-904c-4384-b54e-067f74416078)<br>
As usual, the input is tokenized as a sequence of integers.<br>
The next word is predicted by looking at all the previous ones. But you do not want to have a huge loss in the model just because it's not able to predict the correct ones. That's why you have to use a weighted loss. Instead of averaging the loss for every word in the whole sequence, you weigh the loss for the words within the article with zeros, and those within the summary with ones so the model only focuses on the summary. The cost function is a cross entropy function that ignores the words from the article and thus only sums the losses over the words within the summary. However, when there is little data for the summaries, it actually helps to weigh the article loss with non zero numbers, say 0.2 or 0.5 or even one. That way, the model is able to learn word relationships that are common in the article/input.

At test or inference time, you will input the article with the EOS token to the model and ask for the next word. You will keep asking for the next word until you get a EOS token.<br>
![Screenshot 2024-04-23 at 18 50 15](https://github.com/artainmo/machine-learning/assets/53705599/652fa930-9b7f-4680-875d-e9c7d93570b2)<br>
Note that contrary to supervised learning, here we won't provide the summary as input, only the article.<br>
Transformer models generate probability distributions over all possible words. Sampling from this distribution provides a different summary each time you run the model.

### Week 3: Question Answering
We will build a system that can answer questions. In the real world, such models are usually not built from scratch. This is why we will use existing models, such as BERT, or other transformer models found on the [Hugging Face platform](https://github.com/artainmo/DevOps/tree/main/cloud#general-deep-learning-and-machine-learning), and adapt those models via [transfer learning](https://github.com/artainmo/DevOps/tree/main/cloud#generative-ai-transformers-gpt-self-attention-and-foundation-models). 

You can answer questions either by generating text or by finding text in an existing paragraph. Context-based question answering: takes a question and finds the answer within the given context. Closed book question answering: only takes a question and generates an answer without access to a context. 

#### Transfer learning
Transfer learning adds new learnings to a pre-trained model. This reduces training time, improves predictions because the model learned from different tasks that might help the new task, and it requires less training data.

Two forms of transfer learning exist: feature-based and fine-tuning. With feature-based you learn new word embeddings from one pre-trained model, and you feed those to a new model on a different task. With fine-tuning you use word embeddings from a pre-trained model to fine-tune that same model on a different task. One way to fine-tune is to add a new feedforward layer to the pre-trained model while keeping the previous layers freezed. The new layer can then be fine-tuned.

The more training data, the better the model.<br>
When pre-training, you often use unlabeled data. Unlabeled data is used in self-supervised tasks. An example is taking a phrase, removing a word from it, and trying to predict that missing word.<br>
![Screenshot 2025-05-01 at 16 21 53](https://github.com/user-attachments/assets/dd920170-d244-4b3e-af9d-123738135168)<br>
Instead of masking words, you can also predict the next sentence.

#### Relevant models
Chronological order of different relevant models:<br>
![Screenshot 2025-05-01 at 17 05 27](https://github.com/user-attachments/assets/8cb8b32a-ac80-4645-b06e-948112d67910)

CBOW predicts a word in the middle of windows of fixed length. CBOW is limited by the limited window length and thus captured context. To use the full context ELMo was created using RNNs. GPT uses a transformer decoder only, it is uni-directional whereby you cannot look at the next words but only at the previous ones when predicting a word. Even if ELMo was bi-directional, it was not effective at capturing long-term dependencies compared to transformers. BERT uses a transformer encoder only, and is bi-directional, looking at previous and next words. T5 uses an encoder and decoder which leads to better performance. 




## Resources
* [DeepLearning.AI - Natural Language Processing Specialization: Natural Language Processing with Attention Models](https://www.coursera.org/learn/attention-models-in-nlp)
* [Understanding masking & padding](https://www.tensorflow.org/guide/keras/understanding_masking_and_padding#:~:text=Masking%20is%20a%20way%20to,the%20end%20of%20a%20sequence.)
* [Decoding Strategies that You Need to Know for Response Generation](https://towardsdatascience.com/decoding-strategies-that-you-need-to-know-for-response-generation-ba95ee0faadc)
