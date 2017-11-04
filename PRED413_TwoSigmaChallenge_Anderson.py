import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.tsa as tsa
import math

pd.set_option('display.max_columns', 150)

# load data offline
os.chdir('C:\PythonData')
with pd.HDFStore('train.h5', 'r') as train:
    # Note that the "train" dataframe is the only dataframe in the file
    df = train.get("train")
df.shape
df['y'].describe()

df["y"].hist()
df["y"].hist(bins = 30)
plt.xlabel("Target Variable", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
plt.suptitle("Histogram of Response Variable, y", fontsize=20)
    
# data for modelling
low_y_cut = -0.086093
high_y_cut = 0.093497

y_values_within = ((df['y'] > low_y_cut) & (df['y'] <high_y_cut))

df = df.loc[y_values_within,:]
df.shape

# count of id's per timestamp
fig = plt.figure(figsize=(12, 6))
sns.countplot(x='timestamp', data=df)
plt.title('Number of instruments (ids) per timestamp', fontsize=20)
plt.show()

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

# run ADF test for stationarity in each ID
from statsmodels.tsa.stattools import adfuller
f = pd.DataFrame([])
for x in ID:
    print('%d...' % x)
    cut = df[df.id == x]
    nobs = len(cut)
    t = adfuller(cut.y, autolag ='AIC', maxlag = int(12*(nobs/100)^(1/4)))
    d = {'teststat': t[0],
         'pval': t[1],
         'nlags': t[2],
         'nobs': t[3],
         '5%crit': t[4]['5%'],
         'Nonstationary at 5% level': t[0]>t[4]['5%']}
    f_t = pd.DataFrame(d,index=[x])
    f = f.append(f_t)
print(f)
f['Nonstationary at 5% level'].sum()


# trying a moving average forecasting technique for each id
from sklearn.metrics import r2_score

df = df.sort_values(by=['id', 'timestamp'])
ID = df.id.unique()
#IDx = ID[0:10] (for faster execution)

window = 5
predictions = list()
originals = list()

for x in ID:
    work_df = df[df.id == x].sort_values(by='timestamp', ascending=False)
    train_df = work_df[work_df['timestamp'] < work_df['timestamp'].median()]
    test_df = work_df[work_df['timestamp'] > work_df['timestamp'].median()]
    
    if x == 1056:
        window = 1
        
    if x == 93:
        window = 3
    
    # prepare situation
    train_X = train_df['y'].values
    history = [train_X[i] for i in range(window)]
    test_X = test_df['y'].values
    test = [test_X[i] for i in range(window, len(test_X))]
    #originals.append(test)
    
    # walk forward over time steps in test
    for t in range(len(test)):
        length = len(history)
        yhat = mean([history[i] for i in range(length-window,length)])
        obs = test[t]
        predictions.append(yhat)
        originals.append(obs)
        
R2 = r2_score(originals, predictions)
R = R2 * (sqrt(abs(R2)))
print('Test R: %f' % R)


# trying to build an ARIMA model for each ID
import statsmodels.tsa.api as sta
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf

#df['datetime'] = dt.datetime.fromtimestamp(pd.to_numeric(df['timestamp'], errors='coerce').fillna(0))
#df['datetime'] = pd.to_datetime(df['timestamp'])

work_df = df[['datetime','y']]
work_df = df[df.id == 2158]
ts = pd.Series(work_df['y'].values, index=work_df['datetime'])

nobs = len(ts)
thresh = 1.96/np.sqrt(nobs)
lag_acf = acf(ts, nlags=int(12*(nobs/100)^(1/4)))
lag_pacf = pacf(ts, nlags=int(12*(nobs/100)^(1/4)), method='ols')

#plt.plot(lag_acf)
#plt.axhline(y=0,linestyle='--',color='gray')
#plt.axhline(y=-thresh,linestyle='--',color='gray')
#plt.axhline(y=thresh,linestyle='--',color='gray')
#plt.title('Autocorrelation Function')
#
#plt.plot(lag_pacf)
#plt.axhline(y=0,linestyle='--',color='gray')
#plt.axhline(y=-thresh,linestyle='--',color='gray')
#plt.axhline(y=thresh,linestyle='--',color='gray')
#plt.title('Partial Autocorrelation Function')

acf_cut = np.where(lag_acf > thresh)
acf_cut = list(acf_cut[len(acf_cut)-1])
p = acf_cut[-1]

pacf_cut = np.where(lag_pacf > thresh)
pacf_cut = list(pacf_cut[len(pacf_cut)-1])
q = pacf_cut[-1]

model = ARIMA(ts, order=(p,0,q))
model_fit = model.fit(disp=0)
print(model_fit.summary())


# sandbox
df['diff_y'] = df['y'].shift(1).where(df['id'].shift(1) == df['id'], '')
df.tail()

df['diff_y'] = df.groupby('id')['y'].shift(1).fillna(0)

# differencing? 
#train_data = df.set_index(['id','timestamp']).sort_index()
#ID2=train_data.index.levels[0]

for col in diff_cols:
    # FIXME: why not group by (id, ts)? why only a lag of 1 sample?
    train[col + '_d1'] = train[col].rolling(2).apply(lambda x: x[1] - x[0]).fillna(0)
train = train[train.timestamp != 0] # Drop first timestamp that had no diffs 
#(FIXME can be confusing; why not leave missing?)
