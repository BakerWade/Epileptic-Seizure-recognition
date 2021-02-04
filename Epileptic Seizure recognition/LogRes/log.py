import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data.csv')
df['y'] = df['y'].map({5:0, 4:0, 3:0, 2:0, 1:1})

#sns.histplot(df['y'])
plt.show()

from sklearn.model_selection import train_test_split

x = df.drop(['Unnamed: 0','y'],axis=1)
y = df['y']

x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.3,random_state=101)

from sklearn.linear_model import LogisticRegression

log = LogisticRegression(max_iter=5000)

log.fit(x_train, y_train)

pred = log.predict(x_test)

from sklearn.metrics import r2_score, confusion_matrix, classification_report

print('R2_score:',r2_score(y_test,pred))
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))