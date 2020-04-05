
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import pandas as pd 
import scipy as sp 
import sklearn
import random 
import time 
from sklearn import preprocessing, model_selection
from keras.models import Sequential 
from keras.layers import Dense 
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle


data = pd.read_csv("/home/filip/anaconda3/NeuralNetwork/FeatureVectorsToTrain.csv", sep=',', header=0) # load dataset to train 
#print(data)
#data = data.drop(['Id'], axis=1)
#print(data)
data2 = data;
data = shuffle(data)


#Select 10 test featureVector from whole shuffled dataset.
i = 10
data_to_predict = data[:i].reset_index(drop = True)
predict_species = data_to_predict.Class
predict_species = np.array(predict_species)

prediction = np.array(data_to_predict.drop(['Class'],axis= 1)) # Drop first column as a target classes


data = data2[i:].reset_index(drop = True)

X = data2.drop(['Class'], axis = 1) # set X -> All columns except first Column.
#X.fillna(X.mean(), inplace=True)
X = np.array(X)
Y = data2['Class'] # set first  column "Class" as Y.


# Transform Y to Categorical values
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)
Y = np_utils.to_categorical(Y)


train_x, test_x, train_y, test_y = model_selection.train_test_split(X,Y,test_size = 0.2, random_state = 0)
input_dim = len(data.columns) - 1

#Model
model = Sequential()
model.add(Dense(4096, input_dim = input_dim , activation = 'relu'))
model.add(Dense(5000, activation = 'relu'))
model.add(Dense(5000, activation = 'relu'))
model.add(Dense(100, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )
print("Training...")
model.fit(train_x, train_y, epochs =50, batch_size = 32)


scores = model.evaluate(train_x, train_y, verbose=0);
print("Accuracy on train set: %.2f%%" % (scores[1]*100))

scores = model.evaluate(test_x, test_y, verbose=0);
print("Accuracy on test set: %.2f%%" % (scores[1]*100))

#scores = model.evaluate(validation_x, validation_y, verbose=0);
#print("Accuracy on train set: %.2f%%" % (scores[1]*100))

print("Predicting...")
predictions = model.predict_classes(prediction)

prediction_ = np.argmax(to_categorical(predictions), axis = 1)

prediction_ = encoder.inverse_transform(prediction_)

for i, j in zip(prediction_ , predict_species):
    print( "The NeuralNetwork predict {}, and the Class to find is {}".format(i,j))

model.save('/home/filip/anaconda3/NeuralNetwork/TrainedModel.h5')
