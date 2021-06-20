# -*- coding: utf-8 -*-

"""
Positive or negative comments
"""

import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import joblib
import pandas as pd
import matplotlib.pyplot as plt


labels=['neg', 'pos']
punctuations=['.', ',', "'", '"', '(', ')', ':', '?', '@','!', '~', '/', '<', '>', '#']

data_path = "C:/AQ/Nam_ba/Deep_Learning/RNN/Positive_negative_comment/dataset"

#load data
X_train_full =[]
y_train_full=[]


def getData(name, looking_up_table=None, num_oov_buckets=5, sequence_length=200):
    '''The func is to get data 
    Input: folder name "test" or "train", and looking_up_table for test,..
    Output: X_train_full, y_train_full, vocabulary, looking_up_table'''
    X_train_full=[]
    y_train_full=[]
    
    for folder_name in labels:
        folder_path = os.path.join(data_path, name, folder_name)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            #read data in file
            with open(file_path, 'r') as f:
                try:
                    reviews = f.read()
                    X_train_full.append(reviews)
                    if folder_name=="neg":
                        y_train_full.append(0)
                    else:
                        y_train_full.append(1)                  
                except:
                    print(len(X_train_full))
                    
    #preprocssing data
    #convert to lower case
    X_new=[]       
    for x in X_train_full:
        X_new.append(x.lower())
    
    paragraphs_without_punctuations=[]
    #remove punctuations
    for x in X_new:
        new_paragraph = ''.join([char for char in x if char not in punctuations])
        #print (new_paragraph)
        paragraphs_without_punctuations.append(new_paragraph)
        
    #Split the paragraph into single word to create sequence data for RNN
    list_of_single_word=[]
    for paragraph in paragraphs_without_punctuations:
        list_of_single_word.append(paragraph.split())
    
    #create looking up table for train data, not use for test data
    if looking_up_table ==  None:
        #create a vocalbulary
        all_words=[]
        for i in range(len(list_of_single_word)):
            for j in range(len(list_of_single_word[i])):
                all_words.append(list_of_single_word[i][j])
            
        vocab = np.unique(all_words)
        
        #encoding words in vocab
        indices = tf.range(len(vocab), dtype=tf.int64)
        
        #create lookup table to change words into approriate numbers
        table_init = tf.lookup.KeyValueTensorInitializer(keys=tf.constant(vocab), values=indices)
        
        #Should create additional buckets 
        table = tf.lookup.StaticVocabularyTable(table_init, num_oov_buckets)
    else:
        table= looking_up_table
        vocab=None
        
    # #labels of categories
    # categories= tf.constant(list_of_single_word[0])
    
    # # sample_cats = tf.constant(['last', 'a', 'watched'])
    # print(table.lookup(categories))

    #convert all word to int 
    X_train_full = []
    for paragraph in list_of_single_word:
        X_train_full.append(table.lookup(tf.constant(paragraph)))
        
    #change data into sequence
    X_train_full = pad_feature(X_train_full, sequence_length)
    
    return X_train_full, y_train_full, vocab, table

def changeParagraphToText(comment, looking_up_table, sequence_length=200):
    '''The func is to change text data into int array for test'''
    #remove punctuations
    comment=comment.lower()
    comment_without_punctuations = ''.join([char for char in comment if char not in punctuations])
     
    #split comment to a list of words
    word_list = comment_without_punctuations.split()
    
    #change word into int type
    int_list = looking_up_table.lookup(tf.constant(word_list))
    
    #padding 
    padding_list = pad_feature([int_list], sequence_length)
    return padding_list
    
def pad_feature(sentence_ints, sequence_length=200):
    '''The func is to pad or truncate data 
    Input: sentence_int: an int array which is transfered from orginal sentence
    Ouput: return an array, each element is a time step
    Exp: a b c -> [1,2,3]
    func -> [0, 0, 0, 1,2,3]'''
    
    #
    features = np.zeros(shape=(len(sentence_ints), sequence_length), dtype=int)
    for i, row in enumerate(sentence_ints):
        features[i, -len(row):] = np.array(row)[:sequence_length]
    return features


X_train_full, y_train_full, vocabulary, table = getData("train")
#split data
X_train, X_valid, y_train,y_valid = train_test_split(X_train_full, y_train_full, test_size=0.2)
print("Number of sample: ",len(X_train) )

#create RNN
num_oov_buckets=5
model = keras.Sequential()
model.add(keras.layers.Embedding(input_dim=len(vocabulary)+num_oov_buckets, 
                                 output_dim=2, input_length=200, name="embeded_layer"))
model.add(keras.layers.SimpleRNN(50, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.summary()

#compile model
save_the_best_model_cb = keras.callbacks.ModelCheckpoint("models/sentiment_model.h5")
early_stoping_model_cb = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                       patience=7)


model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(),
            metrics=['accuracy'])

#train model
if 1:
    history = model.fit(np.array(X_train), np.array(y_train), epochs=30,   
                        validation_data=(np.array(X_valid), np.array(y_valid)),
                        callbacks=[save_the_best_model_cb])
    history = history.history
    joblib.dump(history, "models/history")
else:#load model
    model = keras.models.load_model("models/sentiment_model.h5")
    history = joblib.load("models/history")
    
pd.DataFrame(history).plot(ylim=(0,1))
plt.show()

X_train_full, y_train_full, vocabulary, table = getData("train")
#test
if 1:
    model = keras.models.load_model("models/sentiment_model.h5")
    X_test, y_test, vocabulary, table__  = getData("test",looking_up_table=table )
    print("Evaluate model: ", model.evaluate(np.array(X_test), np.array(y_test)))
    
    #test for 1 sample
    # prediction = model.predict(np.array([X_test[0]])).round()
    # print(labels[int(prediction)])
    # print (labels[y_test[0]])
    
    #list prediction results
    for i in range(0, len(X_test), 100):
        print("Predicted value: ",model.predict(np.array([X_test[i]])).round(), ", real value: ", y_test[i] )

#enter a comment and test it
if 1:      
    comment = input("Enter your comment: \n")
    X = changeParagraphToText(comment, table) 
    prediction = model.predict(np.array(X)).round()
    print("The entered comment: '", comment,"'")
    print("Prediction: ", labels[int(prediction)])