This repository contains some approches to information retrieval task.

To run some \*.py modules on restarted machine, we need first to start bert service which in tern needs restored graph in tensorflow.
Graph restoring and service starting are reproduced in the bash script before the main module starts.

We have a __query__ (i.e. "What is better laptop or desktop" and some list of pairs (__title__ - __answer_body__), such as ("Advantages of a laptop over PCs." - "Laptop is preferable than desktop because it is more portable."). The task is to align to every answer a __score__ - digit that describes it closeness ti __query__.

Approcahes:

*3. Approach based on Transformers*

Use *pytorch_transformers* library.

Model: 'bert-base-uncased'

The second output of the forward are the attention's weihgts.

Since the transformer can process 2 sentences are connected by a delimiter [SEP], for every response we create __["[CLS] " + query + " [SEP] " + answer_body + " [SEP]"]__.

Than we apply self-attention and explore the map of the obtained weights. We consider only non-diagonal parts of map are corresponded to weights of word according tokens from __another__ sentence. 

As we can see in **touche-explain_transformers.ipynb**, to count closeness of query and response we need to take into account the 3rd, 4th, 9th and 10 head of attention. The 3rd head highlights the similar words, another heads are responsible for closeness in meaning. 

The score is sum of weights in this head on non-diagonal place, excluding the weights corresponding to the special tokens ([CLS], [SEP], ...).

*5.* Bert embedding + LSTM from FastAi pre-trained language model (AWD_LSTM)
lstm_ulmfit.py, touche_lstm_ulmfit.ipynb
