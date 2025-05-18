def main():
    # Read the text file as separate lines of text
    with open('data.txt', 'r') as file:
        text = file.read()
    lines = text.lower().split('\n')
    # Define words, vocabulary size and sequences of words as lines
    from keras.preprocessing.text import text_to_word_sequence, Tokenizer
    words = text_to_word_sequence(text)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(words)
    vocabulary_size = len(tokenizer.word_index) + 1
    sequences = tokenizer.texts_to_sequences(lines)
    # Find subsequences
    subsequences = []
    for sequence in sequences:
        for i in range(1, len(sequence)):
            subsequence = sequence[:i + 1]
            subsequences.append(subsequence)

    from keras.preprocessing.sequence import pad_sequences
    sequence_length = max([len(sequence) for sequence in sequences])
    sequences = pad_sequences(subsequences, maxlen=sequence_length, padding='pre')

    from keras.utils import to_categorical
    x, y = sequences[:, :-1], sequences[:, -1]
    y = to_categorical(y, num_classes=vocabulary_size)

    from keras.models import Sequential
    model = Sequential()

    from keras.layers import Embedding
    model.add(Embedding(vocabulary_size, 100, input_length=sequence_length - 1))

    from keras.layers import LSTM
    model.add(LSTM(100))

    from keras.layers import Dropout
    model.add(Dropout(0.1))

    from keras.layers import Dense
    model.add(Dense(units=vocabulary_size, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(x, y, epochs=500)

    return model
main()