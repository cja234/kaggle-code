import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os

#%matplotlib inline

pd.set_option('display.max_columns', 120)

os.chdir('C:\PythonData')
with pd.HDFStore('train.h5', 'r') as train:
    # Note that the "train" dataframe is the only dataframe in the file
    df = train.get("train")
df['y'].describe()

# Note that the first observation we get has a "train" dataframe
print("Train has {} rows".format(len(df)))

# The "target" dataframe is a template for what we need to predict:
print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(df.columns)])))

train_data = df
print(df.head())
print(train_data.head())

train_data = train_data.set_index(['id','timestamp']).sort_index()
train_data

train_data.dtypes

f_cols = [f for f in train_data.columns if f.startswith('fundamental')]
t_cols = [t for t in train_data.columns if t.startswith('technical')]
ID = train_data.index.levels[0]

print('The number of fundamental columns is {}.'.format(len(f_cols)))
print('The number of technical columns is {}.'.format(len(t_cols)))
print('The number of unique ids is {}.'.format(len(ID)))

# Exploration of predicted variable, y
df.y.describe()
train_data.y.describe()

fig, axs = plt.subplots(5,2)
font = {'weight' : 'normal',
        'size'   : 10}
plt.rc('font', **font)

rn = ID[np.random.randint(0,len(ID)-1,10)]

for i in range(0,len(rn)):
    ax = plt.subplot(5,2,i+1)
    ax.plot(train_data.loc[rn[i]].index, 
            train_data.y.loc[rn[i]],
            label='ID={}'.format(rn[i]))
    plt.legend()
    if i in [8,9]:
        ax.set_xlabel('Time Stamp')
    if i in range(0,9,2):
        ax.set_ylabel('y')  

# count missing values
labels = []
values = []
for col in df.columns:
    labels.append(col)
    values.append(df[col].isnull().sum())
    print(col, values[-1])

#fill missing values
low_y_cut = -0.086093
high_y_cut = 0.093497

#df = df.sample(frac=0.1)
mean_vals = df.mean(axis=0)
df.fillna(mean_vals, inplace=True)
y_is_within_cut = ((df['y'] > low_y_cut) & (df['y'] < high_y_cut))

#train_cut = df.loc[y_is_within_cut,:]



#train_data = train_data.sample(frac=0.1)
#mean_vals2 = train_data.mean(axis=0, level=1)
#train_data.fillna(mean_vals, inplace=True)
#
#mean_vals
#mean_vals2
#
#
## feature importance
#train_X = df.loc[y_is_within_cut, df.columns[2:-1]]
#train_y = df.loc[y_is_within_cut, 'y'].values.reshape(-1, 1)
#print("Data for model: X={}, y={}".format(train_X.shape, train_y.shape))
#
#import xgboost as xgb
#model = xgb.XGBRegressor()
#print("Fitting...")
#model.fit(train_X, train_y)
#print("Fitting done")
#
#import matplotlib.pyplot as plt
#fig, ax = plt.subplots(figsize=(7, 30))
#xgb.plot_importance(model, ax=ax)
#print("Features importance done")
#
#train_X = train_data.drop('y',axis=1)
#train_Y = train_data.y
#print("Data for model: X={}, y={}".format(train_X.shape, train_Y.shape))
#
#print("Fitting...")
#model.fit(train_X, train_Y)
#print("Fitting done")
#
#fig, ax = plt.subplots(figsize=(7, 30))
#xgb.plot_importance(model, ax=ax)
#print("Features importance done")








# Columns for training
#cols = [col for col in train_data_cleaned.columns if "technical_" in col]
cols = 'technical_20'
print(cols)

low_y_cut = -0.086093
high_y_cut = 0.093497

y_values_within = ((train_data['y'] > low_y_cut) & (train_data['y'] <high_y_cut))

train_cut = train_data.loc[y_values_within,:]

# Fill missing values
#mean_vals = train_cut.mean()
#train_cut.fillna(mean_vals,inplace=True)


import fancyimpute

solver = fancyimpute.MICE(
    n_nearest_columns=11,
    n_imputations=1,
    n_burn_in=0)

# X_incomplete has missing data which is represented with NaN values
train_data = solver.complete(train_data_NA)


train_data = fancyimpute.SoftImpute(max_iters=10).complete(train_data_NA)
train_data = pd.DataFrame(train_data)    

print(train_data_NA.head())
print(train_data.head())


x_train = train_cut[cols]
y = train_cut["y"]

from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()

print(x_train.shape)
print(y.shape)
lr_model.fit(np.array(x_train.values).reshape(-1,1),y.values)
R2 = lr_model.score(np.array(x_train.values).reshape(-1,1),y.values)
print(sqrt(R2))

print('Coefficients: \n', lr_model.coef_)







# kagglegym
#
#while True:
#    observation.features.fillna(mean_vals, inplace=True)
#    x_test = np.array(observation.features[cols].values).reshape(-1,1)
#    ypred = lr_model.predict(x_test)
#    observation.target.y = ypred
#
#    target = observation.target
#    timestamp = observation.features["timestamp"][0]
#    if timestamp % 100 == 0:
#        print("Timestamp #{}".format(timestamp))
#
#    # We perform a "step" by making our prediction and getting back an updated "observation":
#    observation, reward, done, info = env.step(target)
#    if done:
#        print("Public score: {}".format(info["public_score"]))
#        break