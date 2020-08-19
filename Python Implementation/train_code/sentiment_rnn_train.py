import os
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Bidirectional
from keras.preprocessing import sequence
from keras.layers import Dropout
import h5py
import train_code.utility_functions as uf
from keras.models import model_from_json
from keras.models import load_model
from nltk.tokenize import RegexpTokenizer

def load_data_all(data_dir, all_data_path,pred_path, gloveFile, first_run, load_all):

    # Load embeddings for the filtered glove list
    if load_all == True:
        weight_matrix, word_idx = uf.load_embeddings(gloveFile)
    else:
        weight_matrix, word_idx = uf.load_embeddings(filtered_glove_path)

    len(word_idx)
    len(weight_matrix)

    #%%
    # create test, validation and trainng data
    all_data = uf.read_data(all_data_path)
    train_data, test_data, dev_data = uf.training_data_split(all_data, 0.8, data_dir)

    train_data = train_data.reset_index()
    dev_data = dev_data.reset_index()
    test_data = test_data.reset_index()

    #%%
    # inputs from dl_sentiment that are hard coded but need to be automated
    maxSeqLength, avg_words, sequence_length = uf.maxSeqLen(all_data)
    numClasses = 10
    #%%

     # load Training data matrix
    train_x = uf.tf_data_pipeline_nltk(train_data, word_idx, weight_matrix, maxSeqLength)
    test_x = uf.tf_data_pipeline_nltk(test_data, word_idx, weight_matrix, maxSeqLength)
    val_x = uf.tf_data_pipeline_nltk(dev_data, word_idx, weight_matrix, maxSeqLength)

    #%%
    # load labels data matrix
    train_y = uf.labels_matrix(train_data)
    val_y = uf.labels_matrix(dev_data)
    test_y = uf.labels_matrix(test_data)


     #%%

    # summarize size
    print("Training data: ")
    print(train_x.shape)
    print(train_y.shape)

    # Summarize number of classes
    print("Classes: ")
    print(np.unique(train_y.shape[1]))

    return train_x, train_y, test_x, test_y, val_x, val_y, weight_matrix, word_idx

def create_model_rnn(weight_matrix, max_words, EMBEDDING_DIM):

    # create the model
    model = Sequential()
    model.add(Embedding(len(weight_matrix), EMBEDDING_DIM, weights=[weight_matrix], input_length=max_words, trainable=False))
    model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.50))
    model.add(Dense(10, activation='softmax'))
    # try using different optimizers and different optimizer configs
    model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    return model

def train_model(model,train_x, train_y, test_x, test_y, val_x, val_y, batch_size, path) :

    # save the best model and early stopping
    saveBestModel = keras.callbacks.ModelCheckpoint(path+'/model/best_model.hdf5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')

    # Fit the model
    model.fit(train_x, train_y, batch_size=batch_size, epochs=25,validation_data=(val_x, val_y), callbacks=[saveBestModel, earlyStopping])
    # Final evaluation of the model
    score, acc = model.evaluate(test_x, test_y, batch_size=batch_size)

    print('Test score:', score)
    print('Test accuracy:', acc)

    return model

def live_test(trained_model, data, word_idx):

    #data = "Pass the salt"
    #data_sample_list = data.split()
    live_list = []
    live_list_np = np.zeros((56,1))
    # split the sentence into its words and remove any punctuations.
    tokenizer = RegexpTokenizer(r'\w+')
    data_sample_list = tokenizer.tokenize(data)

    labels = np.array(['1','2','3','4','5','6','7','8','9','10'], dtype = "int")
    #word_idx['I']
    # get index for the live stage
    data_index = np.array([word_idx[word.lower()] if word.lower() in word_idx else 0 for word in data_sample_list])
    data_index_np = np.array(data_index)
    print(data_index_np)

    # padded with zeros of length 56 i.e maximum length
    padded_array = np.zeros(56) # use the def maxSeqLen(training_data) function to detemine the padding length for your data
    padded_array[:data_index_np.shape[0]] = data_index_np
    data_index_np_pad = padded_array.astype(int)
    live_list.append(data_index_np_pad)
    live_list_np = np.asarray(live_list)
    type(live_list_np)

    # get score from the model
    score = trained_model.predict(live_list_np, batch_size=1, verbose=0)
    #print (score)

    single_score = np.round(np.argmax(score)/10, decimals=2) # maximum of the array i.e single band

    # weighted score of top 3 bands
    top_3_index = np.argsort(score)[0][-3:]
    top_3_scores = score[0][top_3_index]
    top_3_weights = top_3_scores/np.sum(top_3_scores)
    single_score_dot = np.round(np.dot(top_3_index, top_3_weights)/10, decimals = 2)

    #print (single_score)
    return single_score_dot

def main():

    max_words = 56 # max no of words in your training data
    batch_size = 2000 # batch size for training
    EMBEDDING_DIM = 100 # size of the word embeddings
    train_flag = True # set True if in training mode else False if in prediction mode

    if train_flag:
        # create training, validataion and test data sets
        # load the dataset
        path = '<<Add your code directory path>>'
        data_dir = path+'/Data'
        all_data_path = path+'/Data/'
        pred_path = path+'/Data/output_model/test_pred.csv'
        gloveFile = path+'/Data/glove/glove_6B_100d.txt'
        first_run = False
        load_all = True

        train_x, train_y, test_x, test_y, val_x, val_y, weight_matrix, word_idx = load_data_all(data_dir, all_data_path,pred_path, gloveFile, first_run, load_all)
        # create model strucutre
        model = create_model_rnn(weight_matrix, max_words, EMBEDDING_DIM)

        # train the model
        trained_model =train_model(model,train_x, train_y, test_x, test_y, val_x, val_y, batch_size, path)   # run model live


        # serialize weights to HDF5
        model.save_weights(path+"/model/best_model.h5")
        print("Saved model to disk")

    else:
        gloveFile = path+'/Data/glove/glove_6B_100d.txt'
        weight_matrix, word_idx = uf.load_embeddings(gloveFile)
        weight_path = path +'/model/best_weights_bi_glove.hdf5'
        loaded_model = load_model(weight_path)
        loaded_model.summary()
        data_sample = "Great!! it is raining today!!"
        result = live_test(loaded_model,data_sample, word_idx)
        print (result)
