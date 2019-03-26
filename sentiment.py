from keras.datasets import imdb

vocabulary_size = 5000

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocabulary_size)

print('Loaded dataset with {} training samples, {} test samples'.format(len(X_train),len(X_test)))

print('---review---')
print(X_train[6])
print('---label---')
print(y_train[6])

word2id = imdb.get_word_index()
id2word = {i: word for word, i in word2id.items()}
print('---review with words---')
print([id2word.get(i, '') for i in X_train[6]])
print('--label---')
print(y_train[6])


#padding input
from keras.preprocessing import sequence
max_words = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)


#building the model
from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

embedding_size = 32
model = Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_words))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#traning the model
batch_size = 64
num_epochs = 3
X_valid, y_valid = X_train[:batch_size], y_train[:batch_size]
X_train2, y_train2 = X_train[batch_size:], y_train[batch_size:]

model.fit(X_train,y_train, validation_data=(X_test,y_test), batch_size = batch_size, epochs=num_epochs)

model.save('Sentiment_analysis.h5')

scores = model.evaluate(X_test,y_test, verbose=0)
print('Test accuracy:', scores[1])





