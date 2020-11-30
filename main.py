# DATA 301 Project
# Edward Du, Cal Schwefler, Ethan Choi

import sqlite3
import numpy as np
import pandas as pd
import sklearn.ensemble as ske
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

pd.set_option('display.max_columns', None)
# conn = sqlite3.connect('/content/drive/My Drive/Cal Poly/DATA 301/DATA 301 Project/FPA_FOD_20170508.sqlite')
conn = sqlite3.connect('/content/drive/My Drive/Data301 files/FPA_FOD_20170508.sqlite')
df = pd.read_sql_query("SELECT * FROM Fires", conn)

''' Here we generate some plots to better understand the data we are dealing with '''
''' Determine important features for random forest '''
fire_cause = df['STAT_CAUSE_DESCR'].to_frame()
fire_cause.columns=['cause of fire']
fire_cause = fire_cause.value_counts()
fire_cause.plot.bar()
plt.ylabel=("Number")
plt.title("Number of fires vs Cause of fire")
plt.show()

fire_state = df['STATE'].to_frame()
fire_state.columns=['state']
fire_state = fire_state.value_counts()
fire_state.head(15).plot.bar()
plt.ylabel=("Number")
plt.title("Number of fires per state")
plt.show()

'''Now we will filter unnecessary attributes out of our data as well as begin 
organizing the data into groups based on fire cause that we have established'''
fire_info = pd.DataFrame(np.c_[df['DISCOVERY_DATE'],df['DISCOVERY_DOY'],df['DISCOVERY_TIME'], df['STAT_CAUSE_CODE'], df['STAT_CAUSE_DESCR'],df['CONT_DATE'], \
                               df['CONT_DOY'],df['CONT_TIME'],df['FIRE_SIZE'],df['LATITUDE'],df['LONGITUDE'],df['OWNER_CODE']], \
                         columns= ['date', 'day of year', 'time', 'cause code', 'cause descr', 'containment date', 'containment doy', 'containment time', 'size', 'latitude', \
                                   'longitude', 'owner code'])

fire_info = fire_info.dropna()


grouped_causes = []
unknown = ['Miscellaneous', 'Missing/Undefined'] # 1
unintentional = ['Railroad', 'Powerline', 'Structure'] # 2
preventable = ['Debris Burning', 'Children', 'Equipment Use', 'Campfire', 'Smoking', 'Fireworks'] # 3
natural = ['Lightning'] # 4
purposeful = ['Arson'] # 5
causes_name_list = fire_info['cause descr'].tolist()
for each in causes_name_list:
  if each in unknown:
    grouped_causes.append(1)
  elif each in unintentional:
    grouped_causes.append(2)
  elif each in preventable:
    grouped_causes.append(3)
  elif each in natural:
    grouped_causes.append(4)
  elif each in purposeful:
    grouped_causes.append(5)

fire_causes = pd.DataFrame(np.c_[fire_info['cause code'], fire_info['cause descr']], columns=['cause code', 'cause descr'])
fire_causes['grouped cause code'] = grouped_causes
fire_info = fire_info.drop(['cause code', 'cause descr'], axis=1)

# print(fire_info)
# print(fire_causes)

'''Now that our data is properly formatted we will split it into training, testing, and validation sets'''
X_df = fire_info
y_df = fire_causes['grouped cause code'].values
X = np.asarray(X_df).astype('float32')
y = np.asarray(y_df).astype('float32')

# print(X)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size = 0.25, random_state = 42)

'''Now we will train our model using the random forest algorithm as well as check how well our testing data fits'''
rf = ske.RandomForestClassifier(n_estimators=100)
rf = rf.fit(X_train, y_train)
print(rf.score(X_test,y_test))
pred = rf.predict(X_test)

'''Now we will check our validation data'''
training_pred = rf.predict(X_validate)

'''And now we compute the accuracy of our testing prediction'''
target_names = ['Unknown', 'Unintentional', 'Preventable', 'Natural', 'Purposeful']
conf_mat = confusion_matrix(y_test, pred)
print(classification_report(y_test, pred, target_names=target_names))
print(conf_mat)

'''And finally we will use our validation data to check the accuracy of our validation prediction'''
target_names = ['Unknown', 'Unintentional', 'Preveantable', 'Natural', 'Purposeful']
conf_mat = confusion_matrix(y_validate, training_pred)
print(classification_report(y_validate, training_pred, target_names=target_names))
print(conf_mat)