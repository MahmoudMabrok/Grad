## Get file names ##
## ************** ## 


'''
import os
files_names = os.listdir( os.getcwd() + "/s1" )



## Extract labels from files names ##
## ******************************* ##
labels = [ int(f[2][:3]) for f in files_names ]



## Refactor file names by adding the parent directory name before the file ##
## *********************************************************************** ##
files = [ "s1/" + f for f in files_names ]



## Read audio files and extract MFCC, Delta-MFCC ##
## ********************************************* ##
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank

_mfcc = list()
_delta_mfcc = list()

for f in files:
    (rate, sig) = wav.read( f )

    new_mfcc = mfcc( sig, rate )
    llll =  new_mfcc[ :420, :] 
    _mfcc.append( new_mfcc[ :250, :] )

    _delta_mfcc.append( delta( new_mfcc, 2 )[ :250, ] )   



## Vectorize features ##
## ****************** ## item = matrix || sublist = _mfcc
mfcc_vector_list = []
for matrix in _mfcc:
    l = matrix.tolist()
    flat_list = [item for sublist in l for item in sublist]
    mfcc_vector_list.append( flat_list )

delta_mfcc_vector_list = []
for matrix in _delta_mfcc:
    l = matrix.tolist()
    flat_list = [item for sublist in l for item in sublist]
    delta_mfcc_vector_list.append( flat_list )



## Combine files names, MFCC, Delta-MFCC, and label together ##
## ********************************************** ##
dataset = list()
for i in range( len( files_names ) ):
    dataset.append( [ labels[i] ] )
    dataset[i].extend( mfcc_vector_list[i] )
    dataset[i].extend( delta_mfcc_vector_list[i] )
  

## Write to csv file ##
## ***************** ##
import csv
with open( "dataset.csv", "w" ) as data_file:
    writer = csv.writer( data_file )
    writer.writerows( dataset )

'''

# our model 
import  warnings

warnings.simplefilter('ignore')
 

import pandas as pd

data = pd.read_csv('dataset.csv')

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values


# Feature Selection using RFE 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0) 

#print('s')
#
#from sklearn.feature_selection import RFE
#rfe = RFE(classifier, 5) # 
#fit = rfe.fit(X, y)
#
#ll = list()
#
#s = fit.support_ 
#for i in range(len(s)):
#    b = s[i]
#    if b : ll.append(i)
#
#X = data.iloc[ : , ll ]
#
#print('f')
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm )



from sklearn.metrics import accuracy_score  
print ('accuracy ' , accuracy_score(y_test, y_pred))  # 0.1 (for 0.2) --> 30%(0.3) 


