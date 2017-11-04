import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.tsa as tsa

pd.set_option('display.max_columns', 150)

# load data offline
os.chdir('C:\PythonData')
with pd.HDFStore('train.h5', 'r') as train:
    # Note that the "train" dataframe is the only dataframe in the file
    df = train.get("train")
df.shape
    
# data for modelling
low_y_cut = -0.086093
high_y_cut = 0.093497

y_values_within = ((df['y'] > low_y_cut) & (df['y'] <high_y_cut))

df = df.loc[y_values_within,:]
df.shape

# drop rows that only have values in 5 or fewer columns
df = df.dropna(thresh=6)
df.shape

# sort by ID and timestamp to impute in time series order (if necessary)
df = df.sort_values(by=['id', 'timestamp'])
ID = df.id.unique()

imp_df = pd.DataFrame([])
report0 = 0

for x in ID:
    cut = df[df.id == x]
    before = cut.isnull().sum()
    cut.fillna(method='pad', inplace=True)
    after = cut.isnull().sum()
    report0 = sum(before - after)
    report1 = report1 + report0
    imp_df = imp_df.append(cut)
    
report2 = len(imp_df)
print('%d NaNs replaced... %d rows in imp_df.' % (report1, report2))


work_df = df[df.id == 10]

from statsmodels.tsa.stattools import adfuller
f = pd.DataFrame([])
#for i in rn:
t = adfuller(work_df.y, autolag ='AIC')
d = {'teststat': t[0],
    'pval': t[1],
    'nlags': t[2],
    'nobs': t[3],
    '1%crit': t[4]['1%'],
    '5%crit': t[4]['5%'],
    '10%crit': t[4]['10%'],
    'Nonstationary at 1% level': t[0]>t[4]['1%'],
    'Nonstationary at 5% level': t[0]>t[4]['5%'],
    'Nonstationary at 10% level': t[0]>t[4]['10%']}
f_t = pd.DataFrame(d,index=[10])
f = f.append(f_t)

#train_data = df.set_index(['id','timestamp']).sort_index()
#ID2=train_data.index.levels[0]

for col in diff_cols:
    # FIXME: why not group by (id, ts)? why only a lag of 1 sample?
    train[col + '_d1'] = train[col].rolling(2).apply(lambda x: x[1] - x[0]).fillna(0)
train = train[train.timestamp != 0] # Drop first timestamp that had no diffs 
#(FIXME can be confusing; why not leave missing?)


