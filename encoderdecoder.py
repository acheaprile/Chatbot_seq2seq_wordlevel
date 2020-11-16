import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Model
from keras.layers import Input, LSTM, Dense, TimeDistributed, Embedding, Dropout

# Import the text, shuffle it and add the start and end of sequence characters
questanswcsv="C:\\Users\\Daniel.GarciaPerez\\Desktop\\Python\\github\\chatbot\\enfr.csv"
qatext=pd.read_csv(questanswcsv)
qatext=qatext.head(30000).sample(frac=1)
qatext=qatext.rename(columns={qatext.columns[0]:"questions", qatext.columns[1]:"answers"})
questions=(qatext["questions"]).astype(str)
answers=("SOSEN "+qatext["answers"]+" EOSEN").astype(str)

# Define model training parameters and latent dimension for the LSTMs
epochs=100
bsize=500
valsplit=0.2
hidden_dim=400

# Tokenize the texts
def tokenizer(texts, file):
    
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='!"#$%&()*+,-/:;<=>@?.[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(texts)
    
    # Save a word-index dictionary for inference
    pd.DataFrame(data=[tokenizer.word_index.keys(),tokenizer.word_index.values()]).T.to_csv("{}.csv".format(file))
    
    return (tokenizer)

# Transform each sentence into integer sequences
def get_sequences(texts, tokenizer):
    
    text_sequences=tokenizer.texts_to_sequences(texts)
    unique_words=max([index for word in text_sequences for index in word])+1
    max_seq_len=max([len(text) for text in text_sequences])
    
    sequences = np.array(tf.keras.preprocessing.sequence.pad_sequences(text_sequences, maxlen=max_seq_len, padding="post"))

    
    return (sequences, unique_words, max_seq_len)

def onehotencoding(numwords, answersdecoder):
    
    onehot=np.array(tf.keras.backend.one_hot(indices= range(0,numwords) , num_classes=numwords ))
    onehotmatch=np.array([onehot[answer] for answer in answersdecoder])
    
    return (onehotmatch)

def model(encoder_uniquewords, encoder_max_seqlen, decoder_uniquewords,
          decoder_max_seqlen, encoder_input, decoderinput, decoderoutput, hidden_dim):

    encoder_inputs=(Input(shape=(None, )))
    embedlayer1=Embedding(input_dim=encoder_uniquewords, 
                                output_dim=hidden_dim)(encoder_inputs)
    encoderoutput, state_h, state_c = LSTM(units= hidden_dim ,
                                               return_state=True, name="encoderoutput")(embedlayer1)
    encoder_states=[state_h, state_c]
    
    decoder_inputs = Input(shape=(None, ))
    embedlayer2=Embedding(input_dim=decoder_uniquewords, 
                                output_dim=hidden_dim)(decoder_inputs)
    lstmoutput, _, _ = LSTM(units=hidden_dim,return_sequences=True, return_state=True)(embedlayer2,
                           initial_state=encoder_states)
    denselayer1= Dense(units=1000, activation="relu")(lstmoutput)
    dropout1= Dropout(0.2)(denselayer1)
    softmaxlayer= Dense(units=decoder_uniquewords,activation="softmax")
    decoder_outputs= TimeDistributed(softmaxlayer)(dropout1)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy",
                  metrics=["accuracy"])
    
    model.summary()
    
    callback1=tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15)
    callback2=tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15)
    callback3=tf.keras.callbacks.ModelCheckpoint("encoderdecoder.hdf5", monitor="val_accuracy",
                                                 save_best_only=True)
    
    history=model.fit([encoder_input, decoderinput], decoderoutput, epochs=epochs, batch_size=bsize,
                      validation_split=valsplit, shuffle=True, callbacks=[callback1, callback2, callback3])
    
    return(model, history)

questionsvector=get_sequences(questions, tokenizer(questions, "encoderdict"))
encoder_input=questionsvector[0]
encoder_uniquewords=questionsvector[1]
encoder_max_seqlen=questionsvector[2]

answerstokens=tokenizer(answers, "decoderdict")
answersvectorinput=get_sequences(answers.replace(to_replace=" EOSEN",value="", regex=True), answerstokens)
decoder_input=answersvectorinput[0]
decoder_uniquewords=answersvectorinput[1]
answersvectoroutput=get_sequences(answers.replace(to_replace="SOSEN ",value="", regex=True), answerstokens)
decoder_max_seqlen=answersvectoroutput[2]
decoder_output=onehotencoding(decoder_uniquewords, answersvectoroutput[0])

print ("Number of Samples:", len(qatext))
print ("Unique input tokens:", encoder_uniquewords)
print ("Unique output tokens:", decoder_uniquewords)
print ("Input max length seq:", encoder_max_seqlen)
print ("Output max length seq:", decoder_max_seqlen)

history=model(encoder_uniquewords, encoder_max_seqlen, decoder_uniquewords,
          decoder_max_seqlen, encoder_input, decoder_input, decoder_output, hidden_dim)[1]

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()