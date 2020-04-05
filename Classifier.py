import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
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
from keras.models import load_model


model = load_model('/home/filip/anaconda3/NeuralNetwork/TrainedModel.h5') # load trained model

data = pd.read_csv('/home/filip/anaconda3/NeuralNetwork/FeatureVectorsToCompare.csv', sep=',') # load feature vectors of two fingerprints

file1 = open("/home/filip/anaconda3/NeuralNetwork/results.txt","w") 

prediction = np.array(data)

predictions = model.predict_classes(prediction)

for i in predictions:
	file1.write('%d\n' % i)
	print("The NeuralNetwork classified fingerprint to class:", i)
	
file1.close()
	
	
	
	
	
	
	
