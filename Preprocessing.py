#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer

train = pd.read_csv('train.csv', index_col='id')
n = train.shape[0]
test = pd.read_csv('test.csv', index_col='id')
#print(test)
data = pd.concat([train, test], sort=False)

#print(train.head())
#print(train.shape)

#train['revenue'].describe()
top_count = 21

def list_to_names(s):
    s = eval(s)
    names = []
    size = len(s)    
    for i in range(size):
        names.append(s[i]['name'])
    
    return names

def dummy_names(column):
    global data
    data[column].fillna("[{'name': ''}]", inplace=True)
    data[column] = data[column].apply(list_to_names)
    data['num_' + column] = data[column].apply(lambda x: len(x) if x != {} else 0)
    #print(data[column])
    
    mlb = MultiLabelBinarizer()
    buffer = pd.DataFrame(mlb.fit_transform(data[column]), columns=column + '_' + mlb.classes_, index=data.index)
    data.reset_index(drop=True, inplace=True)
    buffer.reset_index(drop=True, inplace=True)
    
    data = pd.concat([data, buffer], axis=1, sort=False)
    data.drop([column, column + '_'], axis=1, inplace=True)

#----------------column 'belongs_to_collection'-------------

data['belongs_to_collection'].fillna("[{'name': ''}]", inplace=True)
data['belongs_to_collection'] = data['belongs_to_collection'].apply(list_to_names)
data['belongs_to_collection'] = data['belongs_to_collection'].apply(lambda x: ''.join(x))
data['has_collection'] = 0
data['has_collection'].loc[data['belongs_to_collection'] != ''] = 1

#------------column 'genres'-------------------------

dummy_names('genres')

#----------------'production_companies'--------------------------

def top_20_counts(col):
    global data
    data[col].fillna("[{'name': ''}]", inplace=True)
    data[col] = data[col].apply(list_to_names)
    data['num_' + col] = data[col].apply(lambda x: len(x) if x != {} else 0)
    cols = []
    data[col].apply(cols.extend)
    col_values = {}
    for column in cols:
        if column in col_values:
            col_values[column] += 1
        else:
            col_values[column] = 1
    
    top_cols_values = Counter(col_values).most_common(top_count)
    
    top_cols = [c[0] for c in top_cols_values]
    try:
        top_cols.remove('')
    except:
        pass
        
    def drop_col(l):
        res = []
        for column in l:        
            if column in top_cols:
                res.append(column)
        return res
    
    data[col] = data[col].apply(drop_col)
    mlb = MultiLabelBinarizer()
    buffer = pd.DataFrame(mlb.fit_transform(data[col]), columns=col + '_' + mlb.classes_, index=data.index)
    data.reset_index(drop=True, inplace=True)
    buffer.reset_index(drop=True, inplace=True)    
    data = pd.concat([data, buffer], axis=1, sort=False)
    data.drop([col], axis=1, inplace=True)

top_20_counts('production_companies')

#-------------------'production_countries'-------------------------

dummy_names('production_countries')

#----------------------'release date'---------------------------------

def day_of_week(s):
    if s == '0/0/0':
        return 0
    else:
        return datetime.datetime.strptime(s, '%m/%d/%y').weekday() + 1

data['release_date'].fillna('0/0/0', inplace=True)
split_dates = data['release_date'].str.split('/', expand=True)
split_dates.rename(columns={0: 'release_month', 1: 'release_day', 2: 'release_year'}, inplace=True)
data = pd.concat([data, split_dates], axis=1, sort=False)
data['release_year'] = data['release_year'].apply(lambda x: '19'+x if int(x) > 18 else '20'+x)
data['day_of_week'] = data['release_date'].apply(day_of_week)
data.drop('release_date', axis=1, inplace=True)

#-----------------------'spoken_languages'---------------------------

dummy_names('spoken_languages')

#----------------------'Keywords'-----------------------------

top_20_counts('Keywords')

#-------------------------'cast'------------------------------------

top_20_counts('cast')

#-------------------------'crew'------------------------------------

top_20_counts('crew')

#--------------------------------------------------------

data['has_homepage'] = 1
data['has_homepage'].loc[data['homepage'].isna()] = 0

#-----------------------------------------------------------------

data['title_diff'] = 0
data['title_diff'].loc[data['original_title'] != data['title']] = 1

#-------------------------------------------------------------------

data['runtime'].fillna(np.mean(data['runtime']), inplace=True)

#---------------------------------------------------------------------

data['has_tagline'] = 1
data['has_tagline'].loc[data['tagline'].isna()] = 0
data['tagline'].fillna('', inplace=True)
data['tagline_len'] = data['tagline'].apply(lambda s: len(s.split()))
data.drop(['tagline'], axis=1, inplace=True)

#---------------------------------------------------------------------

data['budget'] = np.log1p(data['budget'])
data['popularity'] = np.log1p(data['popularity'])

data.drop(['status', 'belongs_to_collection', 'poster_path', 'homepage', 'imdb_id', 'original_title', 'title', 'overview'], axis=1, inplace=True)

#print(data)

train = data[:n]
train['revenue'] = np.log1p(train['revenue'])
test = data[n:]
test.drop('revenue', axis=1, inplace=True)

train.to_csv('train_last.csv', index=True, index_label='id')
test.to_csv('test_last.csv', index=True, index_label='id')
print('Done')


# In[ ]:




