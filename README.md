# Transformer from scratch using TensorFlow

![15658638](https://github.com/Kirouane-Ayoub/transformer_from_scratch-tensorflow/assets/99510125/f1354e6c-d1b0-48e6-a08c-5a6f15896b9e)

![eAKQu](https://github.com/Kirouane-Ayoub/transformer_from_scratch/assets/99510125/4d676a46-4f31-4cea-91e0-00825aa5730a)


A transformer is a neural network architecture that relies on the parallel multi-head attention mechanism. It was first introduced in the paper "Attention Is All You Need" by Ashish Vaswani et al. in 2017.

Transformers have become the dominant approach for natural language processing (NLP) tasks, such as machine translation, text summarization, and question answering. They have also been used for other tasks, such as image captioning and speech recognition.

The key difference between transformers and other neural network architectures is the use of attention. Attention allows transformers to learn long-range dependencies between different parts of the input sequence. This is important for NLP tasks, where the meaning of a word can depend on the words that come before and after it.

Transformers are also more efficient than other neural network architectures. They can be trained on large datasets using parallel processing, which makes them faster to train.

Here are some of the key components of a transformer:

## Self-attention: 
Self-attention is a mechanism that allows a transformer to learn the relationship between different parts of the input sequence. It does this by computing a weighted sum of the representations of all the other words in the sequence.

## Multihead attention: 
Multihead attention is a generalization of self-attention that allows a transformer to learn multiple relationships between different parts of the input sequence. This is done by computing a weighted sum of the representations of the input sequence, each weighted by a different attention vector.

## Positional embeddings: 
Positional embeddings are used to encode the order of the words in the input sequence. This is important for NLP tasks, where the meaning of a word can depend on its position in the sentence.

## Feedforward neural networks: 
Feedforward neural networks are used to map the output of the attention layers to a final representation of the input sequence.
