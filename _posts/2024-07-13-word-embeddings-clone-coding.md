---
layout: post
category: NLP
---

This post is a word embedding clone coding workshop following **Coding Lane**'s tutorial video. 
Original content can be found [this](https://github.com/Coding-Lane/Training-Word-Embeddings---Scratch) GitHub repository.

## Table of contents
- [Code Flow](#code-flow)
- [Why Biagrams?](#biagram)
- [NN Architecture](#nn)

## [Code Flow](#code-flow)

This is the basic outline of how model is trained to embed words.

1. Remove stop words and tokenize
  - Stop words are words like 'the', 'is', 'are', 'can', etc. That don't lie on the correlationship among different words.
  - For efficiencty, these stop words are cleaned in this tutorial example.

2. Creating Bigrams
  - Bigrams are all possible sets of two different words that appear in a given sentence.

3. One-hot encoding
  - One-hot encoding is a vanilla approach of representing different words.
  - We need one-hot encoded vectors to represent all words for input and output of neural network.

4. Build neural network

  ```
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense, Input
  
  model = Sequential()
  
  vocal_size = len(onehot_data[0])
  embed_size = 2
  
  model.add(Input(shape = (vocal_size,)))
  model.add(Dense(embed_size, activation = 'linear'))
  model.add(Dense(vocal_size, activation = 'softmax'))
  
  model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
  ```

  - Usually, word embedding is not a super complex deep neural network.
  - It is consisted of three layers - an input, one hidden, and an output layer.
  - For the output layer, we use Softmax activation function.

5. Throw data into the NN

  ```
  X = []
  Y = []
  for bi in bigrams:
      X.append(onehot_dict[bi[0]])
      Y.append(onehot_dict[bi[1]])
  
  X = np.array(X)
  Y = np.array(Y)
  
  model.fit(X, Y, epochs = 1000)
  ```

  - X and Y are onehot representation of bigram word set resprectively.

## [Why Biagrams?](#biagram)
## [NN Architecture](#nn)

