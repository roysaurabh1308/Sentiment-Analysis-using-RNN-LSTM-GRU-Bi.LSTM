import pandas as pd
import numpy as np
import tensorflow as tf
import re
import codecs
import os
from nltk.tokenize import RegexpTokenizer
#%%
################################### Paths to Data ########################################################################

path = '/Data/'
gloveFile = '/Data/glove/glove_6B_300d.txt' #'/Users/prajwalshreyas/Desktop/Singularity/Topic modelling/Glove/glove.twitter.27B/glove.twitter.27B.25d.txt'
vocab_path = '/Data/glove/vocab_glove.csv'

#Split Data path
train_data_path ='/Data/TrainingData/train.csv'
val_data_path ='/Data/TrainingData/val.csv'
test_data_path ='/Data/TrainingData/test.csv'

sent_matrix_path ='/Data/inputs_model/sentence_matrix.csv'
sent_matrix_path_val ='/Data/inputs_model/sentence_matrix_val.csv'
sent_matrix_path_test ='/Data/inputs_model/sentence_matrix_test.csv'
sequence_len_path = '/Data/inputs_model/sequence_length.csv'
sequence_len_val_path = '/Data/inputs_model/sequence_length_val.csv'
sequence_len_test_path = '/Data/inputs_model/sequence_length_test.csv'
wordVectors_path = '/Data/inputs_model/wordVectors.csv'
#%%#

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Filtered Vocabulary from Glove document >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def filter_glove(full_glove_path, data_dir):
  vocab = set()
  sentence_path = os.path.join(data_dir,'SOStr.txt')
  filtered_glove_path = os.path.join(data_dir, 'filtered_glove.txt')
  # Download the full set of unlabeled sentences separated by '|'.
  #sentence_path, = download_and_unzip(
    #'http://nlp.stanford.edu/~socherr/', 'stanfordSentimentTreebank.zip',
    #'stanfordSentimentTreebank/SOStr.txt')
  with codecs.open(sentence_path, encoding='utf-8') as f:
    for line in f:
      # Drop the trailing newline and strip backslashes. Split into words.
      vocab.update(line.strip().replace('\\', '').split('|'))
  nread = 0
  nwrote = 0
  with codecs.open(full_glove_path, encoding='utf-8') as f:
    with codecs.open(filtered_glove_path, 'w', encoding='utf-8') as out:
      for line in f:
        nread += 1
        line = line.strip()
        if not line: continue
        if line.split(u' ', 1)[0] in vocab:
          out.write(line + '\n')
          nwrote += 1
  print('read %s lines, wrote %s' % (nread, nwrote))
#%%#

#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Filtered Vocabulary from live cases >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< load embeddings >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def load_embeddings(embedding_path):
  """Loads embedings, returns weight matrix and dict from words to indices."""
  print('loading word embeddings from %s' % embedding_path)
  weight_vectors = []
  word_idx = {}
  with codecs.open(embedding_path, encoding='utf-8') as f:
    for line in f:
      word, vec = line.split(u' ', 1)
      word_idx[word] = len(weight_vectors)
      weight_vectors.append(np.array(vec.split(), dtype=np.float32))
  # Annoying implementation detail; '(' and ')' are replaced by '-LRB-' and
  # '-RRB-' respectively in the parse-trees.
  word_idx[u'-LRB-'] = word_idx.pop(u'(')
  word_idx[u'-RRB-'] = word_idx.pop(u')')
  # Random embedding vector for unknown words.
  weight_vectors.append(np.random.uniform(
      -0.05, 0.05, weight_vectors[0].shape).astype(np.float32))
  return np.stack(weight_vectors), word_idx


# Combine and split the data into train and test
def read_data(path):
    # read dictionary into df
    df_data_sentence = pd.read_table(path + 'dictionary.txt')
    df_data_sentence_processed = df_data_sentence['Phrase|Index'].str.split('|', expand=True)
    df_data_sentence_processed = df_data_sentence_processed.rename(columns={0: 'Phrase', 1: 'phrase_ids'})

    # read sentiment labels into df
    df_data_sentiment = pd.read_table(path + 'sentiment_labels.txt')
    df_data_sentiment_processed = df_data_sentiment['phrase ids|sentiment values'].str.split('|', expand=True)
    df_data_sentiment_processed = df_data_sentiment_processed.rename(columns={0: 'phrase_ids', 1: 'sentiment_values'})


    #combine data frames containing sentence and sentiment
    df_processed_all = df_data_sentence_processed.merge(df_data_sentiment_processed, how='inner', on='phrase_ids')

    return df_processed_all

def training_data_split(all_data, spitPercent, data_dir):

    msk = np.random.rand(len(all_data)) < spitPercent
    train_only = all_data[msk]
    test_and_dev = all_data[~msk]


    msk_test = np.random.rand(len(test_and_dev)) <0.5
    test_only = test_and_dev[msk_test]
    dev_only = test_and_dev[~msk_test]

    dev_only.to_csv(os.path.join(data_dir, 'TrainingData/dev.csv'))
    test_only.to_csv(os.path.join(data_dir, 'TrainingData/test.csv'))
    train_only.to_csv(os.path.join(data_dir, 'TrainingData/train.csv'))

    return train_only, test_only, dev_only
#%%
################################### Glove Vector  ########################################################################
def loadGloveModel(gloveFile):
    print ("Loading Glove Model")
    f = open(gloveFile,'r',encoding='utf-8')
    model = {}
    for line in f:
        try:
            splitLine = line.split()
            word = splitLine[0]
            embedding = [float(val) for val in splitLine[1:]]
            model[word] = embedding
        except:
            print (word)
            continue

    print ("Done.",len(model)," words loaded!")
    return model
#%%


#%%
################################### Create Vocab subset GLove vectors ########################################################################

def word_vec_index(training_data, glove_model):

    sentences = training_data['Phrase'] # get the phrases as a df series
    #sentences = sentences[0:100]
    sentences_concat = sentences.str.cat(sep=' ')
    sentence_words = re.findall(r'\S+', sentences_concat)
    sentence_words_lwr = [x.lower() for x in sentence_words]
    subdict = {word: glove_model[word] for word in glove_model.keys() & sentence_words_lwr}

    vocab_df = pd.DataFrame(subdict)
    vocab_df.to_csv(vocab_path)
    return vocab_df
#%%
################################### Convertdf to list ########################################################################
def word_list(vocab_df):

    wordVectors = vocab_df.values.T.tolist()
    wordVectors_np = np.array(wordVectors)
    wordList = list(vocab_df.columns.values)

    return wordList, wordVectors_np
 #%%
################################### tensorflow data pipeline ########################################################################


def maxSeqLen(training_data):

    total_words = 0
    sequence_length = []
    idx = 0
    for index, row in training_data.iterrows():

        sentence = (row['Phrase'])
        sentence_words = sentence.split(' ')
        len_sentence_words = len(sentence_words)
        total_words = total_words + len_sentence_words

        # get the length of the sequence of each training data
        sequence_length.append(len_sentence_words)

        if idx == 0:
            max_seq_len = len_sentence_words


        if len_sentence_words > max_seq_len:
            max_seq_len = len_sentence_words
        idx = idx + 1

    avg_words = total_words/index

    # convert to numpy array
    sequence_length_np = np.asarray(sequence_length)

    return max_seq_len, avg_words, sequence_length_np

  #%%
def tf_data_pipeline(data, word_idx, weight_matrix, max_seq_len):

    #training_data = training_data[0:50]

    maxSeqLength = max_seq_len #Maximum length of sentence
    no_rows = len(data)
    ids = np.zeros((no_rows, maxSeqLength), dtype='int32')
    # conver keys in dict to lower case
    word_idx_lwr =  {k.lower(): v for k, v in word_idx.items()}
    idx = 0

    for index, row in data.iterrows():


        sentence = (row['Phrase'])
        sentence_words = sentence.split(' ')

        i = 0
        for word in sentence_words:
            #print(index)
            word_lwr = word.lower()
            try:
                #print (word_lwr)
                ids[idx][i] =  word_idx_lwr[word_lwr]

            except Exception as e:
                #print (e)
                #print (word)
                if str(e) == word:
                    ids[idx][i] = 0
                continue
            i = i + 1
        idx = idx + 1
    return ids

  #%%
# create labels matrix for the rnn


def tf_data_pipeline_nltk(data, word_idx, weight_matrix, max_seq_len):

    #training_data = training_data[0:50]

    maxSeqLength = max_seq_len #Maximum length of sentence
    no_rows = len(data)
    ids = np.zeros((no_rows, maxSeqLength), dtype='int32')
    # conver keys in dict to lower case
    word_idx_lwr =  {k.lower(): v for k, v in word_idx.items()}
    idx = 0

    for index, row in data.iterrows():


        sentence = (row['Phrase'])
        #print (sentence)
        tokenizer = RegexpTokenizer(r'\w+')
        sentence_words = tokenizer.tokenize(sentence)
        #print (sentence_words)
        i = 0
        for word in sentence_words:
            #print(index)
            word_lwr = word.lower()
            try:
                #print (word_lwr)
                ids[idx][i] =  word_idx_lwr[word_lwr]

            except Exception as e:
                #print (e)
                #print (word)
                if str(e) == word:
                    ids[idx][i] = 0
                continue
            i = i + 1
        idx = idx + 1

    return ids


def labels_matrix(data):

    labels = data['sentiment_values']

    lables_float = labels.astype(float)

    cats = ['0','1','2','3','4','5','6','7','8','9']
    labels_mult = (lables_float * 10).astype(int)
    dummies = pd.get_dummies(labels_mult, prefix='', prefix_sep='')
    dummies = dummies.T.reindex(cats).T.fillna(0)
    labels_matrix = dummies.as_matrix()

    return labels_matrix


def labels_matrix_unmod(data):

    labels = data['sentiment_values']

    lables_float = labels.astype(float)

    labels_mult = (lables_float * 10).astype(int)
    labels_matrix = labels_mult.as_matrix()

    return labels_matrix

#%%
################################### Run Steps ########################################################################
def main():

    # Load the Trainign data
    all_data = read_data(path)
    #%%
    training_data = pd.read_csv(train_data_path, encoding='iso-8859-1')

    # use the below to split the training, validation and test
    train_df = training_data_split(training_data)
    #%%

    # Load glove vector
    glove_model = filter_glove(gloveFile)

    # Get glove vector subset for training vocab
    vocab_df = word_vec_index(all_data, glove_model)
    glove_model = None

    #Run this after the first iteration of obtaining the vocab df instead of above 2 steps
    vocab_df = pd.read_csv(vocab_path, encoding='iso-8859-1')

    #Get Wordlist and word vec lists from the df for the training Vocab
    wordList, wordVectors = word_list(vocab_df)
    wordVectors_df = pd.DataFrame(wordVectors)
    wordVectors_df.to_csv(wordVectors_path)

    # get the index of the word vec for each sentences to be input to the tf algo
    max_seq_len, avg_len, sequence_length = maxSeqLen(training_data)
    sequence_length_df = pd.DataFrame(sequence_length)
    sequence_length_df.to_csv(sequence_len_path)

    # training data input matrix
    sentence_matrix = tf_data_pipeline(training_data, wordList, wordVectors, max_seq_len)

    # export the sentence matrix to a csv file for easy load for next iterations
    sentence_matrix_df = pd.DataFrame(sentence_matrix)
    sentence_matrix_df.to_csv(sent_matrix_path)

    #################################################################### validation data set ############################################################
    # load validation data
    val_data = pd.read_csv(val_data_path, encoding='iso-8859-1')

    # load glove model and generat vocab for validation data
    glove_model = loadGloveModel(gloveFile)
    vocab_df_val = word_vec_index(val_data, glove_model)
    glove_model = None
    wordList_val, wordVectors_val = word_list(vocab_df_val)

    # get max length for val data
    max_seq_len_val, avg_len_val, sequence_length_val = maxSeqLen(val_data)
    sequence_length_val_df = pd.DataFrame(sequence_length_val)
    sequence_length_val_df.to_csv(sequence_len_val_path)

    # get the id matrix for val data
    sentence_matrix_val = tf_data_pipeline(val_data, wordList_val, wordVectors_val, max_seq_len)

    # write the val dat to csv
    sentence_matrix_df_val = pd.DataFrame(sentence_matrix_val)
    sentence_matrix_df_val.to_csv(sent_matrix_path_val)

    #################################################################### Test data set ############################################################
    # load test data
    test_data = pd.read_csv(test_data_path, encoding='iso-8859-1')

    # load glove model and generat vocab for test data
    glove_model = loadGloveModel(gloveFile)
    vocab_df_test = word_vec_index(val_data, glove_model)
    glove_model = None
    wordList_test, wordVectors_test = word_list(vocab_df_test)

    # get max length for test data
    max_seq_len_test, avg_len_test, sequence_length_test = maxSeqLen(test_data)
    sequence_length_test_df = pd.DataFrame(sequence_length_test)
    sequence_length_test_df.to_csv(sequence_len_test_path)

    # get the id matrix for test data
    sentence_matrix_test = tf_data_pipeline(test_data, wordList_test, wordVectors_test, max_seq_len_test)

    # write the test dat to csv
    sentence_matrix_df_test= pd.DataFrame(sentence_matrix_test)
    sentence_matrix_df_test.to_csv(sent_matrix_path_test)
