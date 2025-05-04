# Transformers have become the standard model for NLP, similar to convolutional models in computer vision. And all started with Attention!

# In practice, you'll rarely train a transformer model from scratch. Transformers tend to be very large, so they take time, money, and lots of data to train fully. Instead, you'll want to start with a pre-trained model and fine-tune it with your dataset if you need to.

# Hugging Face (🤗) is the best resource for pre-trained transformers. Their open-source libraries simplify downloading and using transformer models like BERT, T5, and GPT-2. And the best part, you can use them alongside either TensorFlow, PyTorch or Flax.

# In this notebook, you'll use 🤗 transformers to use the DistilBERT model for question answering.

# The transformers library provides pipelines for popular tasks like sentiment analysis, summarization, and text generation. A pipeline consists of a tokenizer, a model, and the model configuration. All these are packaged together into an easy-to-use object. Hugging Face makes life easier.
# Pipelines are intended to be used without fine-tuning and will often be immediately helpful in your projects. For example, transformers provides a pipeline for question answering that you can directly use to answer your questions if you give some context. Let's see how to do just that.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from transformers import pipeline

question_answerer = pipeline(task="question-answering", model="distilbert-base-cased-distilled-squad")

# The pipeline 'question_answerer' you just created needs you to pass the question and context as strings. It returns an answer to the question from the context you provided.

context = """
Tea is an aromatic beverage prepared by pouring hot or boiling water over cured or fresh leaves of Camellia sinensis,
an evergreen shrub native to China and East Asia. After water, it is the most widely consumed drink in the world.
There are many different types of tea; some, like Chinese greens and Darjeeling, have a cooling, slightly bitter,
and astringent flavour, while others have vastly different profiles that include sweet, nutty, floral, or grassy
notes. Tea has a stimulating effect in humans primarily due to its caffeine content.

The tea plant originated in the region encompassing today's Southwest China, Tibet, north Myanmar and Northeast India,
where it was used as a medicinal drink by various ethnic groups. An early credible record of tea drinking dates to
the 3rd century AD, in a medical text written by Hua Tuo. It was popularised as a recreational drink during the
Chinese Tang dynasty, and tea drinking spread to other East Asian countries. Portuguese priests and merchants
introduced it to Europe during the 16th century. During the 17th century, drinking tea became fashionable among the
English, who started to plant tea on a large scale in India.

The term herbal tea refers to drinks not made from Camellia sinensis: infusions of fruit, leaves, or other plant
parts, such as steeps of rosehip, chamomile, or rooibos. These may be called tisanes or herbal infusions to prevent
confusion with 'tea' made from the tea plant.
"""

result = question_answerer(question="Where is tea native to?", context=context)

print(result['answer'])
# Output: China and East Asia

# You can also pass multiple questions to your pipeline within a list

questions = ["Where is tea native to?",
                          "When was tea discovered?",
                          "What is the species name for tea?"]

results = question_answerer(question=questions, context=context)

for q, r in zip(questions, results):
        print(f"{q} \n>> {r['answer']}")
"""
Output:
    Where is tea native to? 
    >> China and East Asia
    When was tea discovered? 
    >> 3rd century AD
    What is the species name for tea? 
    >> Camellia sinensis
"""

# Although the models used in the Hugging Face pipelines generally give outstanding results, sometimes you will have particular examples where they don't perform so well. Let's use the following example with a context string about the Golden Age of Comic Books:

context = """
The Golden Age of Comic Books describes an era of American comic books from the
late 1930s to circa 1950. During this time, modern comic books were first published
and rapidly increased in popularity. The superhero archetype was created and many
well-known characters were introduced, including Superman, Batman, Captain Marvel
(later known as SHAZAM!), Captain America, and Wonder Woman.
Between 1939 and 1941 Detective Comics and its sister company, All-American Publications,
introduced popular superheroes such as Batman and Robin, Wonder Woman, the Flash,
Green Lantern, Doctor Fate, the Atom, Hawkman, Green Arrow and Aquaman.[7] Timely Comics,
the 1940s predecessor of Marvel Comics, had million-selling titles featuring the Human Torch,
the Sub-Mariner, and Captain America.[8]
As comic books grew in popularity, publishers began launching titles that expanded
into a variety of genres. Dell Comics' non-superhero characters (particularly the
licensed Walt Disney animated-character comics) outsold the superhero comics of the day.[12]
The publisher featured licensed movie and literary characters such as Mickey Mouse, Donald Duck,
Roy Rogers and Tarzan.[13] It was during this era that noted Donald Duck writer-artist
Carl Barks rose to prominence.[14] Additionally, MLJ's introduction of Archie Andrews
in Pep Comics #22 (December 1941) gave rise to teen humor comics,[15] with the Archie
Andrews character remaining in print well into the 21st century.[16]
At the same time in Canada, American comic books were prohibited importation under
the War Exchange Conservation Act[17] which restricted the importation of non-essential
goods. As a result, a domestic publishing industry flourished during the duration
of the war which were collectively informally called the Canadian Whites.
The educational comic book Dagwood Splits the Atom used characters from the comic
strip Blondie.[18] According to historian Michael A. Amundson, appealing comic-book
characters helped ease young readers' fear of nuclear war and neutralize anxiety
about the questions posed by atomic power.[19] It was during this period that long-running
humor comics debuted, including EC's Mad and Carl Barks' Uncle Scrooge in Dell's Four
Color Comics (both in 1952).[20][21]
"""

question = "What popular superheroes were introduced between 1939 and 1941?"

result = question_answerer(question=question, context=context)
print(result['answer'])
# Output: teen humor comics

# Here, the answer should be: "Batman and Robin, Wonder Woman, the Flash, Green Lantern, Doctor Fate, the Atom, Hawkman, Green Arrow, and Aquaman". Instead, the pipeline returned a different answer. You can even try different question wordings and you will only get incorrect answers.

# The example that fooled your question_answerer belongs to the TyDi QA dataset, a dataset from Google for question/answering in diverse languages. To achieve better results when you know that the pipeline isn't working as it should, you need to consider fine-tuning your model.

# We will fine-tune the model to give better answers for that type of context. To do that, we will be using the TyDi QA dataset but on a filtered version. Additionally, you will use a lot of the tools that Hugging Face has to offer.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments

from sklearn.metrics import f1_score

# As you saw previously, you can use these pipelines as they are. But sometimes, you'll need something more specific to your problem, or maybe you need it to perform better on your production data. In these cases, you'll need to fine-tune a model.

# To get the dataset to fine-tune your model, you will use Hugging Face Datasets, a lightweight and extensible library to share and access datasets and evaluation metrics for NLP easily.
# A common approach is to use 'load_dataset' and get the full dataset but for this lab you will use a filtered version containing only the English examples, which is already saved in the jupyter environment. Since this filtered dataset is saved using the Apache Arrow format, you can read it by using the 'load_from_disk' function.

#The path where the dataset is stored
path = './tydiqa_data/'

#Load Dataset
tydiqa_data = load_from_disk(path)

# The question answering model predicts a start and endpoint in the context to extract as the answer. That's why this NLP task is known as extractive question answering. To train your model, you need to pass start and endpoints as labels. The dataset contains unanswerable questions. For these, the start and end indices for the answer are equal to -1.

# Selecting a subset of the train dataset
flattened_train_data = flattened_train_data.select(range(3000))

# Selecting a subset of the test dataset
flattened_test_data = flattened_test_data.select(range(1000))

# When loading a tokenizer with any method, you must pass the model checkpoint that you want to fine-tune. Here, you are using the'distilbert-base-cased-distilled-squad' checkpoint.

# Import the AutoTokenizer from the transformers library
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")

# Define max length of sequences in the tokenizer
tokenizer.model_max_length = 512

# Tokenizing and processing the flattened dataset
processed_train_data = flattened_train_data.map(process_samples)
processed_test_data = flattened_test_data.map(process_samples)

# You will use the same model from the question-answering pipeline that you loaded in previously.

# Import the AutoModelForQuestionAnswering for the pre-trained model. You will only fine tune the head of the model
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")

# Now, you can take the necessary columns from the datasets to train/test and return them as Pytorch Tensors.
columns_to_return = ['input_ids','attention_mask', 'start_positions', 'end_positions']
processed_train_data.set_format(type='pt', columns=columns_to_return)
processed_test_data.set_format(type='pt', columns=columns_to_return)

#Now, you will use the Hugging Face trainer object to fine-tune your model.

# Training hyperparameters
training_args = TrainingArguments(
    output_dir='model_results',     # output directory
    overwrite_output_dir=True,
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=8,   # batch size per device during training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=20,                 # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_steps=50
)

# Trainer object
trainer = Trainer(
    model=model,                        # the instantiated 🤗 Transformers model to be trained
    args=training_args,                 # training arguments, defined above
    train_dataset=processed_train_data, # training dataset
    eval_dataset=processed_test_data,   # evaluation dataset
    compute_metrics=compute_f1_metrics
)

# Training loop
trainer.train()

# Evaluate performance
trainer.evaluate(processed_test_data)

# After training and evaluating your fine-tuned model, you can check its results for the same previous question.

text = r"""
The Golden Age of Comic Books describes an era of American comic books from the
late 1930s to circa 1950. During this time, modern comic books were first published
and rapidly increased in popularity. The superhero archetype was created and many
well-known characters were introduced, including Superman, Batman, Captain Marvel
(later known as SHAZAM!), Captain America, and Wonder Woman.
Between 1939 and 1941 Detective Comics and its sister company, All-American Publications,
introduced popular superheroes such as Batman and Robin, Wonder Woman, the Flash,
Green Lantern, Doctor Fate, the Atom, Hawkman, Green Arrow and Aquaman.[7] Timely Comics,
the 1940s predecessor of Marvel Comics, had million-selling titles featuring the Human Torch,
the Sub-Mariner, and Captain America.[8]
As comic books grew in popularity, publishers began launching titles that expanded
into a variety of genres. Dell Comics' non-superhero characters (particularly the
licensed Walt Disney animated-character comics) outsold the superhero comics of the day.[12]
The publisher featured licensed movie and literary characters such as Mickey Mouse, Donald Duck,
Roy Rogers and Tarzan.[13] It was during this era that noted Donald Duck writer-artist
Carl Barks rose to prominence.[14] Additionally, MLJ's introduction of Archie Andrews
in Pep Comics #22 (December 1941) gave rise to teen humor comics,[15] with the Archie
Andrews character remaining in print well into the 21st century.[16]
At the same time in Canada, American comic books were prohibited importation under
the War Exchange Conservation Act[17] which restricted the importation of non-essential
goods. As a result, a domestic publishing industry flourished during the duration
of the war which were collectively informally called the Canadian Whites.
The educational comic book Dagwood Splits the Atom used characters from the comic
strip Blondie.[18] According to historian Michael A. Amundson, appealing comic-book
characters helped ease young readers' fear of nuclear war and neutralize anxiety
about the questions posed by atomic power.[19] It was during this period that long-running
humor comics debuted, including EC's Mad and Carl Barks' Uncle Scrooge in Dell's Four
Color Comics (both in 1952).[20][21]
"""

questions = ["What superheroes were introduced between 1939 and 1941 by Detective Comics and its sister company?",
				"What comic book characters were created between 1939 and 1941?",
			    "What well-known characters were created between 1939 and 1941?",
			    "What well-known superheroes were introduced between 1939 and 1941 by Detective Comics?"]

for question in questions:
	inputs = tokenizer.encode_plus(question, text, return_tensors="pt")

	input_ids = inputs["input_ids"].tolist()[0]
	inputs.to("cuda")

	text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
	answer_model = model(**inputs)

	start_logits = answer_model['start_logits'].cpu().detach().numpy()

	answer_start = np.argmax(start_logits)  

	end_logits = answer_model['end_logits'].cpu().detach().numpy()

	answer_end = np.argmax(end_logits) + 1  # Get the most likely end of answer with the argmax of the score

	answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))

	print(f"Question: {question}")
	print(f"Answer: {answer}\n")

# This provides the correct answers thanks to fine-tuning


