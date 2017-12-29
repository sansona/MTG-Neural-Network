import itertools
import numpy as np
import pandas as pd

#loads parameters
df = pd.read_csv('DeckParameters.csv', header=None)

#strips first row corresponding to categories
df = df.drop(df.index[0])
values = df.values

#df = df.drop('Commander', 1)

#database as matrix
#df_matrix = df.as_matrix()
#unravelled matrix
#df_unravelled = df_matrix.ravel().reshape((df_matrix.shape[0]*df_matrix.shape[1],1))


#load raw training data
training = pd.read_csv('TrainingData3.csv')
training = training.drop('Date',1)

#convert training data to matrix & unroll
train_mat = training.as_matrix()
train_vec_raw = train_mat.ravel()

#initialize list of classes (winners) and training examples. Every 3 examples in train_vec correspond to 1 winner in winners
winners = []
train_vec = []

#separates train_vec_raw into list of input data (train_vec) and winners (classif)
for i in range(train_vec_raw.size):
    #%4 == 3 due to intiial formatting of match data - every 4th point corresponds to winner
    if i%4 == 3:
        winners.append(train_vec_raw[i])
    else:
        train_vec.append(train_vec_raw[i])

'''
print ('This is the training vector\n')
print(train_vec)
print ('This is the winners vector\n')
print(winners)
'''
print ('The number of elements in the raw data set is ' + str(len(train_vec_raw)) + '. The number of elements in the training vector are ' + str(len(train_vec)) + '. The number of elements in the winners vector is ' + str(len(winners)) + '.\n')


#sets each element of train_vec equal to the corresponding stats from values
for i in range(len(train_vec)):
    #substitutes commander name with corresponding parameters
    train_vec[i] = (values[values[:, 0] == str(train_vec[i])]).tolist()
    #removes first element (commander name) from list
    del train_vec[i][0][0]

#converts nparray to single list containing floats of all parameters of all decks - ugly, but functional
vectorized_parameters = [float(n) for n in list(itertools.chain.from_iterable(
    list(itertools.chain.from_iterable(train_vec))))]

#print (vectorized_parameters)
#print (len(vectorized_parameters))

#need to find way to remove first element (name of deck) and convert all to ints
#also want to convert this to a generalized function that operates based off the .csv-s loaded into the method name
