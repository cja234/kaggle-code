import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os

pd.set_option('display.max_columns', 120)
pd.set_option('display.max_rows', 120)

os.chdir('C:\PythonData')
with pd.HDFStore('train.h5', 'r') as train:
    # Note that the "train" dataframe is the only dataframe in the file
    df = train.get("train")
df.shape
    
# Note that the first observation we get has a "train" dataframe
print("Train has {} rows".format(len(df)))

# The "target" dataframe is a template for what we need to predict:
print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(df.columns)])))

f_cols = [f for f in df.columns if f.startswith('fundamental')]
t_cols = [t for t in df.columns if t.startswith('technical')]
ID = df.id.unique()

print('The number of fundamental columns is {}.'.format(len(f_cols)))
print('The number of technical columns is {}.'.format(len(t_cols)))
print('The number of unique ids is {}.'.format(len(ID)))


    
#temp_df = df.sort_values(['id', 'timestamp'])    
#temp_df = df.groupby(['id', 'timestamp'])    
#temp_df.head()
#temp_df.tail()
#temp_df.mean()

# data for modelling
low_y_cut = -0.086093
high_y_cut = 0.093497

y_values_within = ((df['y'] > low_y_cut) & (df['y'] <high_y_cut))

df = df.loc[y_values_within,:]
df.shape

## count missing values
#labels = []
#values = []
#for col in df.columns:
#    labels.append(col)
#    values.append(df[col].isnull().sum())
#    print(col, values[-1])

dfStats = df[['y', 'id']].groupby('id').agg([np.median, np.std, np.min, np.max, np.mean]).reset_index()
#dfStats = df.groupby('id').agg(np.mean).reset_index()
##dfStats.sort_values( ('y', 'median'), inplace=True )
print(dfStats.head())
#print(dfStats['y'].apply(np.median))

#df1 = df
#df1.fillna(dfStats, inplace=True)

df = df.dropna(thresh=6)
df.shape

#before = df.isnull().sum()
#
#
#df[df.id == 0]

df = df.sort_values(by=['id', 'timestamp'])
ID = df.id.unique()

imp_df = []
imp_df = pd.DataFrame(imp_df)

for x in ID:
    cut = df[df.id == x]
    before = cut.isnull().sum()
    cut.fillna(method='pad', inplace=True)
    after = cut.isnull().sum()
    report1 = sum(before - after)
    imp_df = imp_df.append(cut)
    report2 = len(imp_df)
    print('%s missing values imputed for ID %s. %s rows appended to DF.' % (report1, x, report2))
    
    
    total = (total + report1)

total



#df_cut = df.loc[cut,:]
#mean_vals = df_cut.mean()
#mean_vals = df[df.id == 10].mean()
#df[df.id == 10].fillna(mean_vals, inplace=True)
#mean_vals = pd.DataFrame(mean_vals)






df1[df1.id == 10].fillna(method='pad', inplace=True)



sum(before - after)


df1 = df.set_index(['id']).sort_index()


# recount missing values
labels = []
values = []
for col in df.columns:
    labels.append(col)
    values.append(df[col].isnull().sum())
    print(col, values[-1])


df2.fillna(method='pad', inplace=True)

# Fill missing values
mean_vals = train_cut.mean()




train_data = df.set_index(['id','timestamp']).sort_index()
train_data.head()

train_data = train_data.reset_index()
train_data.head()


#rf = RandomForestClassifier(n_estimators=100)
#rf.fit(train, target)



add_na_indicators = True
add_diff_features = True
na_indicator_cols = ['technical_9', 'technical_0', 'technical_32', 'technical_16', 
    'technical_38', 'technical_44', 'technical_20', 'technical_30', 'technical_13'] 
    # allegedly selected by tree-based algorithms
diff_cols = ['technical_22', 'technical_20', 'technical_30', 'technical_13', 
    'technical_34'] # also allegedly selected by tree-based algorithms
univar_rlm_cols = ['technical_22', 'technical_20', 'technical_30_d1', 'technical_20_d1',
    'technical_30', 'technical_13', 'technical_34']

        
train_median = train.median(axis = 0)

print('Adding missing value counts per row')
train['nr_missing'] = train.isnull().sum(axis = 1)

print('Adding missing value indicators')
if add_na_indicators:
    for col in na_indicator_cols:
        train[col + '_isna'] = pd.isnull(train[col]).apply(lambda x: 1 if x else 0)
        if len(train[col + '_isna'].unique()) == 1:
            print('Dropped constant missingness indicator:', col, '_isna')
            del train[col + '_isna']
            na_indicator_cols.remove(col)

print('Adding diff features')
if add_diff_features:
    train = train.sort_values(by = ['id', 'timestamp'])
    for col in diff_cols:
        # FIXME: why not group by (id, ts)? why only a lag of 1 sample?
        train[col + '_d1'] = train[col].rolling(2).apply(lambda x: x[1] - x[0]).fillna(0)
    train = train[train.timestamp != 0] # Drop first timestamp that had no diffs 
    #(FIXME can be confusing; why not leave missing?)

# We're going to use all of these features for modeling
base_features = [x for x in train.columns if x not in ['id', 'timestamp', 'y']]    


    