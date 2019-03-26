# LSTM-Sentiment-Analysis
Sentiment analysis can be thought of as the exercise of taking a sentence, paragraph, document, or any piece of natural language, and determining whether that text's emotional tone is positive, negative or neutral. Sentiment Analysis is a common NLP task that Data Scientists need to perform.

## Data Overview
For this analysis we’ll be using a dataset of 50,000 movie reviews taken from IMDb. The data was compiled by Andrew Maas and can be found here: <http://ai.stanford.edu/~amaas/data/sentiment//>

The data is split evenly with 25k reviews intended for training and 25k for testing your classifier. Moreover, each set has 12.5k positive and 12.5k negative reviews.

IMDb lets users rate movies on a scale from 1 to 10. To label these reviews the curator of the data labeled anything with ≤ 4 stars as negative and anything with ≥ 7 stars as positive. Reviews with 5 or 6 stars were left out.

## Recurrent Neural Network
Recurrent Neural Networks are the state of the art algorithm for sequential data and among others used by Apples Siri and Googles Voice Search. This is because it is the first algorithm that remembers its input, due to an internal memory, which makes it perfectly suited for Machine Learning problems that involve sequential data. It is one of the algorithms behind the scenes of the amazing achievements of Deep Learning in the past few years.In a RNN, the information cycles through a loop. When it makes a decision, it takes into consideration the current input and also what it has learned from the inputs it received previously. The two images below illustrate the difference in the information flow between a RNN and a Feed-Forward Neural Network. ![picture alt](https://cdn-images-1.medium.com/max/1000/0*mRHhGAbsKaJPbT21.png)
Another good way to illustrate the concept of a RNN’s memory is to explain it with an example:

Imagine you have a normal feed-forward neural network and give it the word „neuron“ as an input and it processes the word character by character. At the time it reaches the character „r“, it has already forgotten about „n“, „e“ and „u“, which makes it almost impossible for this type of neural network to predict what character would come next.

A Recurrent Neural Network is able to remember exactly that, because of it’s internal memory. It produces output, copies that output and loops it back into the network.

## Long Short-Term Memory
Long Short-Term Memory (LSTM) networks are an extension for recurrent neural networks, which basically extends their memory. Therefore it is well suited to learn from important experiences that have very long time lags in between.

The units of an LSTM are used as building units for the layers of a RNN, which is then often called an LSTM network.

LSTM’s enable RNN’s to remember their inputs over a long period of time. This is because LSTM’s contain their information in a memory, that is much like the memory of a computer because the LSTM can read, write and delete information from its memory.
In an LSTM you have three gates: input, forget and output gate. These gates determine whether or not to let new input in (input gate), delete the information because it isn’t important (forget gate) or to let it impact the output at the current time step (output gate). You can see an illustration of a RNN with its three gates below:
![picture alt](https://cdn-images-1.medium.com/max/1000/0*YEVLdwY6verYMBEa.png)
The gates in a LSTM are analog, in the form of sigmoids, meaning that they range from 0 to 1. The fact that they are analog, enables them to do backpropagation with it.

The problematic issues of vanishing gradients is solved through LSTM because it keeps the gradients steep enough and therefore the training relatively short and the accuracy high.

### Train Accuracy ~ 84%
