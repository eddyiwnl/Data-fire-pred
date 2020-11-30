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
# swapped out for cal to work, just comment mine and uncomment yours
conn = sqlite3.connect('/content/drive/My Drive/Data301 files/FPA_FOD_20170508.sqlite')



df = pd.read_sql_query("SELECT * FROM Fires", conn)

print(df)

''' Determine important features for random forest '''
fire_cause = df['STAT_CAUSE_DESCR'].to_frame()
fire_cause.columns=['cause of fire']
fire_cause = fire_cause.value_counts()
print(fire_cause)
fire_cause.plot.bar()
plt.ylabel=("Number")
plt.title("Number of fires vs Cause of fire")
plt.show()

fire_state = df['STATE'].to_frame()
fire_state.columns=['state']
fire_state = fire_state.value_counts()
# print(fire_state)
fire_state.head(15).plot.bar()
plt.ylabel=("Number")
plt.title("Number of fires per state")
plt.show()


# print(df)
''' below fire info contains string data that needs to be encoded '''
# fire_info = pd.DataFrame(np.c_[df['SOURCE_REPORTING_UNIT'],df['FIRE_YEAR'],df['DISCOVERY_DATE'],df['DISCOVERY_DOY'],df['DISCOVERY_TIME'], df['STAT_CAUSE_CODE'], df['STAT_CAUSE_DESCR'],df['CONT_DATE'], \
#                                df['CONT_DOY'],df['CONT_TIME'],df['FIRE_SIZE'],df['FIRE_SIZE_CLASS'],df['LATITUDE'],df['LONGITUDE'],df['OWNER_CODE'],df['STATE'],df['COUNTY']], \
#                          columns= ['reporting unit', 'year', 'date', 'day of year', 'time', 'cause code', 'cause descr', 'containment date', 'containment doy', 'containment time', 'size', 'size class', 'latitude', \
#                                    'longitude', 'owner code', 'state', 'county'])

# fire_info = pd.DataFrame(np.c_[df['FIRE_YEAR'],df['DISCOVERY_DATE'],df['DISCOVERY_DOY'],df['DISCOVERY_TIME'], df['STAT_CAUSE_CODE'], df['STAT_CAUSE_DESCR'],df['CONT_DATE'], \
#                                df['CONT_DOY'],df['CONT_TIME'],df['FIRE_SIZE'],df['LATITUDE'],df['LONGITUDE'],df['OWNER_CODE']], \
#                          columns= ['year', 'date', 'day of year', 'time', 'cause code', 'cause descr', 'containment date', 'containment doy', 'containment time', 'size', 'latitude', \
#                                    'longitude', 'owner code'])

# draft 2
fire_info = pd.DataFrame(np.c_[df['DISCOVERY_DATE'],df['DISCOVERY_DOY'],df['DISCOVERY_TIME'], df['STAT_CAUSE_CODE'], df['STAT_CAUSE_DESCR'],df['CONT_DATE'], \
                               df['CONT_DOY'],df['CONT_TIME'],df['FIRE_SIZE'],df['LATITUDE'],df['LONGITUDE'],df['OWNER_CODE']], \
                         columns= ['date', 'day of year', 'time', 'cause code', 'cause descr', 'containment date', 'containment doy', 'containment time', 'size', 'latitude', \
                                   'longitude', 'owner code'])

fire_info = fire_info.dropna()

# Perhaps too many different causes
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

print(fire_info)
print(fire_causes)


X_df = fire_info
y_df = fire_causes['grouped cause code'].values
print(y_df)
X = np.asarray(X_df).astype('float32')
y = np.asarray(y_df).astype('float32')

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size = 0.25, random_state = 42)
rf = ske.RandomForestClassifier(n_estimators=100)
rf = rf.fit(X_train, y_train)
print(rf.score(X_test,y_test))
pred = rf.predict(X_test)
training_pred = rf.predict(X_validate)
target_names = ['Unknown', 'Unintentional', 'Preventable', 'Natural', 'Purposeful']
conf_mat = confusion_matrix(y_validate, training_pred)
print(classification_report(y_validate, training_pred, target_names=target_names))
print(conf_mat)


target_names = ['Unknown', 'Unintentional', 'Preventable', 'Natural', 'Purposeful']
conf_mat = confusion_matrix(y_test, pred)
print(classification_report(y_test, pred, target_names=target_names))
print(conf_mat)

print(rf.feature_importances_)

fire_features = fire_info.columns.to_list()
plt.barh(fire_features, rf.feature_importances_)

# Printing residuals
############################### DO NOT RUN THIS FOR DRAFT 2... USE NEXT CODE BLOCK INSTEAD ##########################################
actual = y_test
residual = actual-pred
right1 = 0
wrong1 = 0
right2 = 0
wrong2 = 0
right3 = 0
wrong3 = 0
right4 = 0
wrong4 = 0
right5 = 0
wrong5 = 0

for val in range(len(y_test)):
  if (actual[val]-pred[val]) == 0: # the value is correct
    if actual[val] == 1:
      right1+=1
    if actual[val] == 2:
      right2+=1
    if actual[val] == 3:
      right3+=1
    if actual[val] == 4:
      right4+=1
    if actual[val] == 5:
      right5+=1
  else:
    if actual[val] == 1:
      wrong1+=1
    if actual[val] == 2:
      wrong2+=1
    if actual[val] == 3:
      wrong3+=1
    if actual[val] == 4:
      wrong4+=1
    if actual[val] == 5:
      wrong5+=1



print(len(residual))

plt.figure(figsize=(12,8),dpi=80)
residual_plot = plt.scatter(y_test, residual, s=0.1)
plt.xlabel('actual values')
plt.title('Residual vs actual values')
plt.show()

group1vals = pred[actual==1]
group2vals = pred[actual==2]
group3vals = pred[actual==3]
group4vals = pred[actual==4]
group5vals = pred[actual==5]
plt.figure(1)
plt.hist(group1vals,bins =5)
plt.figure(2)
plt.hist(group2vals,bins =5)
plt.figure(3)
plt.hist(group3vals,bins =5)
plt.figure(4)
plt.hist(group4vals,bins =5)
plt.figure(5)
plt.hist(group5vals,bins =5)
data = [['group1','right',right1,],['group1','wrong',wrong1],
        ['group2','right',right2,],['group2','wrong',wrong2],
        ['group3','right',right3,],['group3','wrong',wrong3],
        ['group4','right',right4,],['group4','wrong',wrong4],
        ['group5','right',right5,],['group5','wrong',wrong5]]
newdf = pd.DataFrame(data,columns = ['grouping','correct','value'])
# print(newdf)
right = [right1,right2,right3,right4,right5]
wrong = [wrong1,wrong2,wrong3,wrong4,wrong5]

plotdata = {'right':right,'wrong':wrong}
newdf = pd.DataFrame(plotdata)
newdf.plot(kind='bar')
plt.show()
# df.plot(kind='bar')
# plt.show()
# Printing residuals
actual = y_test
residual = actual-pred

print(len(residual))

plt.figure(figsize=(12,8),dpi=80)
residual_plot = plt.scatter(y_test, residual, s=0.1)
plt.xlabel('actual values')
plt.title('Residual vs actual values')
plt.show()


