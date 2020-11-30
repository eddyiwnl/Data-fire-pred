import sqlite3
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


pd.set_option('display.max_columns', None)

conn = sqlite3.connect('/content/drive/My Drive/Cal Poly/DATA 301/DATA 301 Project/FPA_FOD_20170508.sqlite')

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

fire_info = pd.DataFrame(np.c_[df['FIRE_YEAR'],df['DISCOVERY_DATE'],df['DISCOVERY_DOY'],df['DISCOVERY_TIME'], df['STAT_CAUSE_CODE'], df['STAT_CAUSE_DESCR'],df['CONT_DATE'], \
                               df['CONT_DOY'],df['CONT_TIME'],df['FIRE_SIZE'],df['LATITUDE'],df['LONGITUDE'],df['OWNER_CODE']], \
                         columns= ['year', 'date', 'day of year', 'time', 'cause code', 'cause descr', 'containment date', 'containment doy', 'containment time', 'size', 'latitude', \
                                   'longitude', 'owner code'])

fire_info = fire_info.dropna()

fire_causes = pd.DataFrame(np.c_[fire_info['cause code'], fire_info['cause descr']], columns=['cause code', 'cause descr'])
fire_info = fire_info.drop(['cause code', 'cause descr'], axis=1)

print(fire_info)
print(fire_causes)



X_df = fire_info
y_df = fire_causes['cause code']

X = np.asarray(X_df).astype('float32')
y = np.asarray(y_df).astype('float32')

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size = 0.25, random_state = 42)

rf = RandomForestClassifier(n_estimators=100)
rf = rf.fit(X_train, y_train)
print(rf.score(X_test,y_test))
pred = rf.predict(X_test)
print('Training Score: ', rf.score(X_train, y_train), 'Validation Score: ', rf.score(X_validate, y_validate))

print('Testing Score: ', rf.score(X_test, y_test), 'Validation Score: ', rf.score(X_validate, y_validate))

# from sklearn import tree, preprocessing
# from sklearn.ensemble import RandomForestRegressor

# # 50 estimators took around 8-10 minutes to run, started with 100 (recommended), but since data is currently 900k rows, cut it down to 50
# # tried setting max_depth to 6 (currently is set at unlimited, it runs into leaves are pure), but accuracy dropped
# # have yet to play with the data so we can definitely fine-tune this
# rf = RandomForestRegressor(n_estimators=50)

# rf = rf.fit(X_train, y_train)
# pred = rf.predict(X_test)
# print(rf.score(X_test,y_test))

target_names = ['Lightning', 'Equipment Use', 'Smoking', 'Campfire', \
                'Debris Burning', 'Railroad', 'Arson', 'Children', \
                'Miscellaneous', 'Fireworks', 'Powerline', 'Structure', \
                'Missing/Undefined']
conf_mat = confusion_matrix(y_test, pred)
print(classification_report(y_test, pred, target_names=target_names))
print(conf_mat)

# Printing residuals
actual = y_test
residual = actual-pred

print(len(residual))

plt.figure(figsize=(12,8),dpi=80)
residual_plot = plt.scatter(y_test, residual, s=0.1)
plt.xlabel('actual values')
plt.title('Residual vs actual values')
plt.show()
accuracy_table = pd.DataFrame(actual)
accuracy_table.columns = ['cause of fire']
accuracy_table['prediction'] = pred
accuracy_table['rounded pred'] = np.round(pred)
i = 0
accuracy_table['hit or miss'] = False
print(accuracy_table)
for index, row in accuracy_table.iterrows():
  # print(row['cause of fire'], row['rounded pred'])
  if (row['cause of fire']  == row['rounded pred']):
    # print("EQUAL")
    accuracy_table.iloc[i,accuracy_table.columns.get_loc('hit or miss')] = True
  # accuracy_table.iloc[i]['hit or miss'] = False
  i+=1
# accuracy_table['hit or miss'] = accuracy_table['rounded pred'].equals(accuracy_table['cause of fire'])

total_count_table = accuracy_table
total_count_table = total_count_table.groupby(['cause of fire'])['rounded pred'].count().reset_index()
total_count_table.columns=['cause of fire', 'total guesses']
print(total_count_table)

# .count() gives the TOTAL number of guesses per index
# .sum() gives the number of correct guesses
print(accuracy_table)
accuracy_table = accuracy_table.groupby(['cause of fire'])['hit or miss'].sum().reset_index()
print(accuracy_table)


# test1 = 1.0
# test2 = 2.0
