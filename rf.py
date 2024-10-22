import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, SelectFpr, SelectFromModel

from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.preprocessing import QuantileTransformer

from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import RBFSampler
import copy

import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood
import torch.nn.functional as F

import torch.nn as nn
import torch.optim as optim

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def preprocess_german(df, preprocess):
    # String to int for all attributes. Follow original code
    df['status'] = df['status'].map({'A11': 0, 'A12': 1, 'A13': 2, 'A14': 3}).astype(int) 
    df['savings'] = df['savings'].map({'A61': 0, 'A62': 1, 'A63': 2, 'A64': 3, 'A65': 4}).astype(int)

    df['credit_hist'] = df['credit_hist'].map({'A34': 0, 'A33': 1, 'A32': 2, 'A31': 3, 'A30': 4}).astype(int) # string -> int
    df['employment'] = df['employment'].map({'A71': 0, 'A72': 1, 'A73': 2, 'A74': 3, 'A75': 4}).astype(int)    
    df['gender'] = df['personal_status'].map({'A91': 1, 'A92': 0, 'A93': 1, 'A94': 1, 'A95': 0}).astype(int)
    df['job'] = df['job'].map({'A171': 0, 'A172': 1, 'A173': 2, 'A174': 3}).astype(int)
    df['install_plans'] = df['install_plans'].map({'A141': 1, 'A142': 1, 'A143': 0}).astype(int)
    
    if 'debtors' in df.columns: 
        df['debtors'] = df['debtors'].map({'A101': 0, 'A102': 1, 'A103': 2}).astype(int)     
    if 'property' in df.columns:
        df['property'] = df['property'].map({'A121': 3, 'A122': 2, 'A123': 1, 'A124': 0}).astype(int)        
    if 'telephone' in df.columns:
        df['telephone'] = df['telephone'].map({'A191': 0, 'A192': 1}).astype(int)
    if 'foreign_worker' in df.columns:
        df['foreign_worker'] = df['foreign_worker'].map({'A201': 1, 'A202': 0}).astype(int)

    if preprocess:
        # TODO: this part can also be trained!
        #df = pd.concat([df, pd.get_dummies(df['purpose'], prefix='purpose')],axis=1) # filtered out next
        #df = pd.concat([df, pd.get_dummies(df['housing'], prefix='housing')],axis=1)
        df.loc[(df['credit_amt'] <= 2000), 'credit_amt'] = 0 # df.loc[cond, col]=newvalue
        df.loc[(df['credit_amt'] > 2000) & (df['credit_amt'] <= 5000), 'credit_amt'] = 1
        df.loc[(df['credit_amt'] > 5000), 'credit_amt'] = 2    
        df.loc[(df['duration'] <= 12), 'duration'] = 0
        df.loc[(df['duration'] > 12) & (df['duration'] <= 24), 'duration'] = 1
        df.loc[(df['duration'] > 24) & (df['duration'] <= 36), 'duration'] = 2
        df.loc[(df['duration'] > 36), 'duration'] = 3
        df['age'] = df['age'].apply(lambda x : 1 if x >= 45 else 0) # 1 if old, 0 if young
    
    df = df.drop(columns=['credit_hist', 'purpose', 'credit_amt', 'savings', 'employment', 'age', 'install_plans',\
            'housing', 'num_credits', 'job', 'num_liable', 'personal_status','credit']) 

    return df

# preprocess data
cols = ['status', 'duration', 'credit_hist', 'purpose', 'credit_amt', 'savings', 'employment',\
            'install_rate', 'personal_status', 'debtors', 'residence', 'property', 'age', 'install_plans',\
            'housing', 'num_credits', 'job', 'num_liable', 'telephone', 'foreign_worker', 'credit'] # raw cols
df = pd.read_table('german.data', names=cols, sep=" ", index_col=False)
subset_cols=['status', 'duration', 'credit_hist', 'purpose', 'credit_amt', 'savings', 'employment', 'age', 'install_plans',\
            'housing', 'num_credits', 'job', 'num_liable', 'personal_status','credit'] # filtered cols
df=df[subset_cols]
df['credit'] = df['credit'].replace(2, 0) # 1 Good, 2 Bad credit risk. binary classification: 0 bad 1 good
y = df['credit']
df = preprocess_german(df, True)


# 20% data to train original model
df_train_ori, df_rest, y_train_ori, y_rest = train_test_split(df, y, test_size=0.8, random_state=12) 
# train ori model
classifier = LogisticRegression() 
classifier.fit(df_train_ori, y_train_ori) 
print(f"Ori model train: {classifier.score(df_train_ori, y_train_ori):.4f} Ori model eval: {classifier.score(df_rest, y_rest):.4f}") 


def get_sample_points(df, y, classifier, nslice=400, nsample=100):
    rbf_lst = [] # X for training
    acc_lst = [] # y for training
    for i in range(nslice):
        df_slice = df.sample(n=nsample, random_state=i) 
        y_slice = y[df_slice.index]
        rbf_feature = RBFSampler(gamma=1, random_state=1, n_components=5) 
        rbf_slice = rbf_feature.fit_transform(df_slice) 

        acc = classifier.score(df_slice, y_slice)
        acc_lst.append(acc)
        rbf_lst.append(rbf_slice)
    return rbf_lst, acc_lst

# 40% data to train, 40% data to eval
df_train, df_eval, y_train, y_eval = train_test_split(df_rest, y_rest, test_size=0.5, random_state=12) 
rbf_lst_train, acc_lst_train = get_sample_points(df_train, y_train, classifier)
rbf_lst_eval, acc_lst_eval = get_sample_points(df_eval, y_eval, classifier)

# train RandomForest model
train_x = np.stack([np.mean(vector, axis=0) for vector in rbf_lst_train], axis=0)  # [nslice, nsample, nrbf] -> [nslice, nrbf]

# scaler is not necessary to train RandomForest!
#scaler = StandardScaler()
#train_x = scaler.fit_transform(train_x)
train_y = np.array(acc_lst_train)

# Initialize RandomForest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(train_x, train_y)

# start eval!
# eval on train
pred_train = rf_model.predict(train_x)
true_train = train_y
max_pred_train_index = np.argmax(pred_train)
min_pred_train_index = np.argmin(pred_train)
max_train_x = train_x[max_pred_train_index]
min_train_x = train_x[min_pred_train_index]
print(f"train pred max: {max_train_x.tolist()}")
print(f"train pred min: {min_train_x.tolist()}")

max_true_train_index = np.argmax(true_train)
min_true_train_index = np.argmin(true_train)
max_true_train_x = train_x[max_true_train_index]
min_true_train_x = train_x[min_true_train_index]
print(f"train true max: {max_true_train_x.tolist()}")
print(f"train true min: {min_true_train_x.tolist()}")

# eval on eval
val_x = np.stack([np.mean(vector, axis=0) for vector in rbf_lst_eval], axis=0)  # [nslice, nsample, nrbf] -> [nslice, nrbf]
#val_x = scaler.transform(val_x)

pred = rf_model.predict(val_x)
true = np.array(acc_lst_eval)
max_pred_index = np.argmax(pred)
min_pred_index = np.argmin(pred)
max_x = val_x[max_pred_index]
min_x = val_x[min_pred_index]
print(f"eval pred max: {max_x.tolist()}")
print(f"eval pred min: {min_x.tolist()}")

max_true_index = np.argmax(true)
min_true_index = np.argmin(true)
max_true_x = val_x[max_true_index]
min_true_x = val_x[min_true_index]
print(f"eval true max: {max_true_x.tolist()}")
print(f"eval true min: {min_true_x.tolist()}")

# visualize val
fig, (ax5, ax6) = plt.subplots(1, 2, figsize=(16, 6))
ax5.scatter(pred, true, marker='o', color='tab:green', label='Pred vs True')
min_val = min(ax5.get_xlim()[0], ax5.get_ylim()[0])
max_val = max(ax5.get_xlim()[1], ax5.get_ylim()[1])
ax5.plot([min_val, max_val], [min_val, max_val], color='blue', linestyle='--', label='y = x')
ax5.set_xlabel('Prediction')
ax5.set_ylabel('True Values')
ax5.grid(True)
ax5.legend(loc='upper left')
ax5.set_title('Val')

# visualize train
ax6.scatter(pred_train, true_train, marker='o', color='tab:green', label='Pred vs True')
min_val = min(ax6.get_xlim()[0], ax6.get_ylim()[0])
max_val = max(ax6.get_xlim()[1], ax6.get_ylim()[1])
ax6.plot([min_val, max_val], [min_val, max_val], color='blue', linestyle='--', label='y = x')
ax6.set_xlabel('Prediction')
ax6.set_ylabel('True Values')
ax6.grid(True)
ax6.legend(loc='upper left')
ax6.set_title('Train')

plt.tight_layout()
plt.show()
plt.savefig("rf-nhid5.pdf", format="pdf")