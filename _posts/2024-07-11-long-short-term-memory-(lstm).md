---
layout: post
category: RNN
---

LSTM and GRU algorithms are used in stated of the art in deep learning applications such as speech recongnition, speech synthesis, natural language understanding, etc.

## Table of contents
- [Purpose](#purpose)
- [Idea](#idea)
- [Activation Functions](#activation-functions)
- [Architecture](#architecture)
- [Example](#example)

## [Purpose](#purpose)

LSTM is advanced algorithm of vanilla Recurrent Neural Network, developed to avoid exploding and vanishing gradient problem of RNN. RNN is good for processing sequential data for predictions but suffer from short-term memory. LSTM and GRU are created as a method to mitigate short-term memory using mechanisms called **gates**. Gates are just neural network that regulate the flow of imformation being passed from one time step to the next.[^1]

## [Idea](#idea)

LSTM uses two seperate paths to make predictions.

- Long-term memory
- Short-term memory

## [Activation Functions](#activation-functions)

- Sigmoid: Any x-axis coordinate map into y-axis coordinate between **0 and 1**.
  
  ![Sigmoid activation f](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/1200px-Logistic-curve.svg.png)

- tanh: Any x-axis coordinate map into y-axis coordinate between **-1 and 1**.
  
  ![tanh activation f](https://vidyasheela.com/web-contents/img/post_img/39/tanh%20activation%20function-new.png)

## [Cell Architecture](#architecture)

![LSTM Cell Architceture Figure](https://miro.medium.com/v2/resize:fit:984/1*Mb_L_slY9rjMr8-IADHvwg.png)
*LSTM Cell Architceture Figure*[^2]

- **Long Term Memory**
  - No weights or biases
  - Prevent gradient from exploding or vanishing
- **Short Term Memory**
  - Directly connected to weights
 
### Stages of a unit

#### Stage 1. (f_t)
- The first stage in a LSTM unit determines what percentage of the Long Term Memory is remembered.
- Any miscellaneous weights (smaller than 0) will turn into numbers close to 0 (because of sigmoid activation fuction) and the total multiplication result among weights will output 0.
- Weights closer to 0 means *to forget* and closer to 1 means *to keep*.
- That's why this stage is also called as "Forget Gate". It produces **forget vector**.

#### Stage 2. (i_t)
- This stage determines how we should update the Long Term Memory.
- Consisted of two activation functions: sigmoid and tanh function.
  - Sigmoid f: Transform values between 0 and 1. **Percentage(%) of potential memory to be remembered**.
  - Tanh f: Squish values between -1 and 1. **Potential long term memory**.
- By adding forget vector with candidate, calculate **new cell state**.
- This stage provides new input weights that produces new cell state. That's why it is also called as "Input Gate".

#### Stage 3. (o_t)
- The final stage calculates output from entire LSTM unit. Decides what next hidden state should be.
- Plug previous hidden state to sigmoid function that decides **percentage(%) of potential memory to be remembered**.
- Plug new cell state to tanh function that decides **potential short term memory**.
- Multiply sigmoid(hidden state) with tanh(new cell state) and produce **new short term memory**.
- This is the output of an entire LSTM unit. That's why the stage is also called as "Output Gate"

## [Code Example](#exmaple)

Python seudo code examlpe of a LSTM cell[^3]:

```
def LSTMCell(prev_ct, prev_ht, input):
  combine = prev_ht + input
  ft = forget_layer(combine) # removes non-relevent data
  candidate = candidate_layer(combine) # holds possible values to be held in the cell state
  it = input_layer(combine) # what data from a candidate layer should be eventually added to the new cell state
  Ct = prev_ct * ft + candidate * it # new cell state
  ot = output_layer(combine) # output is calculated
  ht = ot * tanh(Ct) # finally gives new hidden state as a final output of a LSTM cell
  return ht, Ct # return new hidden state and cell state

ct = [0, 0, 0]
gt = [0, 0, 0]

for input in inputs:
  ct, ht = LSTMCell(ct, ht, input)
```

---
{: data-content="footnotes"}

[^1]: Script from *[this video](https://www.youtube.com/watch?v=8HyCNIVRbSU)*
[^2]: Figure from *[this article](https://medium.com/@ottaviocalzone/an-intuitive-explanation-of-lstm-a035eb6ab42c)*
[^3]: Seudo code from *[this video](https://www.youtube.com/watch?v=8HyCNIVRbSU)*
