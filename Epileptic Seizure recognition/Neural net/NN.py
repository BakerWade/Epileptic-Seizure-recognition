import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.python.keras.backend import dropout

df = pd.read_csv('data.csv')
df['y'] = df['y'].map({5:0, 4:0, 3:0, 2:0, 1:1})

from sklearn.model_selection import train_test_split

x = df.drop(['Unnamed: 0','y'],axis=1)
y = df['y']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.callbacks import EarlyStopping
from keras.models import load_model

model = Sequential()

model.add(Dense(178, activation='relu'))
#model.add(Dropout())

model.add(Dense(356, activation='relu'))
#model.add(Dropout(0.5))

model.add(Dense(356, activation='relu'))
#model.add(Dropout(0.5))

#model.add(Dense(356, activation='relu'))
#model.add(Dropout(0.5))

model.add(Dense(178, activation='relu'))
#model.add(Dropout())

model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')
early = EarlyStopping(monitor='val_loss', mode='min',verbose=1,patience=25)

model.fit(x=X_train,y=y_train,epochs=500, validation_data=(X_test,y_test),verbose=2)

model.save('Epileptic_seizure2.keras')
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()
plt.show()

pred = model.predict_classes(X_test)

from sklearn.metrics import classification_report, confusion_matrix,r2_score

print('R2_score:',r2_score(y_test,pred))
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
