import numpy as np
import pandas as pd
import opendatasets as od
from sklearn import preprocessing


# Preparing dataset
od.download("https://www.kaggle.com/datasets/subhamyadav580/dating-site")

all_data = pd.read_csv('./dating-site/profiles.csv')
all2_data = all_data.loc[:,['age', 'body_type', 'diet', 'drinks', 'drugs', 'education',
       'ethnicity', 'height', 'income', 'job',
       'last_online', 'location', 'offspring', 'orientation', 'pets',
       'religion', 'sex', 'sign', 'smokes', 'speaks', 'status']]
drop_data = all2_data.dropna(axis=0,how='any')
cata_data = drop_data.loc[:,['age', 'body_type', 'diet', 'drinks', 'drugs', 'education',
       'ethnicity', 'height', 'income', 'job',
       'last_online', 'location', 'offspring', 'orientation', 'pets',
       'religion', 'sex', 'sign', 'smokes', 'speaks', 'status']]


# Preprocessing
le = preprocessing.LabelEncoder()


cata_data['body_type'] = le.fit_transform(cata_data['body_type'])
cata_data['diet'] = le.fit_transform(cata_data['diet'])
cata_data['drinks'] = le.fit_transform(cata_data['drinks'])
cata_data['drugs'] = le.fit_transform(cata_data['drugs'])
cata_data['education'] = le.fit_transform(cata_data['education'])
cata_data['ethnicity'] = le.fit_transform(cata_data['ethnicity'])
cata_data['job'] = le.fit_transform(cata_data['job'])
cata_data['last_online'] = le.fit_transform(cata_data['last_online'])
cata_data['location'] = le.fit_transform(cata_data['location'])
cata_data['offspring'] = le.fit_transform(cata_data['offspring'])
cata_data['orientation'] = le.fit_transform(cata_data['orientation'])
cata_data['pets'] = le.fit_transform(cata_data['pets'])
cata_data['religion'] = le.fit_transform(cata_data['religion'])
cata_data['sex'] = le.fit_transform(cata_data['sex'])
cata_data['sign'] = le.fit_transform(cata_data['sign'])
cata_data['smokes'] = le.fit_transform(cata_data['smokes'])
cata_data['speaks'] = le.fit_transform(cata_data['speaks'])
cata_data['status'] = le.fit_transform(cata_data['status'])

scaler = preprocessing.StandardScaler()

for i in cata_data.columns:
    
    x = cata_data[i].values.reshape(-1, 1)
    cata_data[i] = scaler.fit_transform(x)

data1 = cata_data[:100]
data2 = cata_data[100:7100]
userID = []
itemID = []
ratings = []
for i in data1.index:
    for j in data2.index:
        userID = np.append(userID, i)
        itemID = np.append(itemID, j)
        
        test1 = data1.loc[i]
        test2 = data2.loc[j]
        
        vector1 = np.array(test1)
        vector2 = np.array(test2)
        
        rating = np.linalg.norm(vector1-vector2)
        ratings = np.append(ratings, rating)

df = pd.DataFrame({'ratings': ratings})
x = df['ratings'].values.reshape(-1, 1) #returns a numpy array
x_scaled = scaler.fit_transform(x)
df['ratings'] = x_scaled

rate = pd.DataFrame({'userID':userID, 'itemID':itemID, 'ratings':df['ratings']})

# Saving csv files
rate.to_csv('ratings.csv')

essays = all_data.loc[:,['essay0','essay1','essay2','essay3','essay4',
                        'essay5','essay6','essay7','essay8','essay9']]

user = drop_data[:100]
user = user.join(essays)
item = drop_data[100:7100]
item = item.join(essays)


user.to_csv('users.csv')
item.to_csv('items.csv')

