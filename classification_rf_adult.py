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
from sklearn.preprocessing import OrdinalEncoder
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
from sklearn.metrics import confusion_matrix
from sklearn.compose import ColumnTransformer
from imblearn.under_sampling import RandomUnderSampler

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 
                'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 
                'hours_per_week', 'native_country', 'income']

data_path = 'adult.data'
train_df = pd.read_csv(data_path, names=column_names, sep=',\s*', engine='python', na_values='?')
print(train_df.head())
print(f"original N: {train_df.shape[0]}")
train_df = train_df.dropna()
print(f"cleaned N: {train_df.shape[0]}")

test_data_path = 'adult.test'
test_df = pd.read_csv(test_data_path, names=column_names, sep=',\s*', engine='python', na_values='?', skiprows=1)
test_df['income'] = test_df['income'].str.strip('.')
print(f"original N: {test_df.shape[0]}")
test_df = test_df.dropna()
print(f"cleaned N: {test_df.shape[0]}")

# preprocess adult
train_df['income'] = train_df['income'].apply(lambda x: 1 if '>50K' in x else 0)
test_df['income'] = test_df['income'].apply(lambda x: 1 if '>50K' in x else 0)
X_train = train_df.drop('income', axis=1)
y_train = train_df['income']
X_test = test_df.drop('income', axis=1)
y_test = test_df['income']

numeric_features = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
categorical_features = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')), 
    ('scaler', StandardScaler()) 
])
#onehot
#categorical_transformer = Pipeline(steps=[
#    ('imputer', SimpleImputer(strategy='most_frequent')), 
#    ('onehot', OneHotEncoder(handle_unknown='ignore')) 
#])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder()) 
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

X_train_preprocessed = pipeline.fit_transform(X_train)
X_test_preprocessed = pipeline.transform(X_test)
#print(X_train_preprocessed.toarray()[0]) # the boundary of one hot vectors is not clear
print(X_train_preprocessed[0]) 

###########data preparing###########
all_columns = numeric_features + categorical_features
df = pd.DataFrame(X_train_preprocessed, columns=all_columns, index=y_train.index)
y = y_train
#rus = RandomUnderSampler(random_state=42)
#df, y = rus.fit_resample(df, y)
####################################

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
        rbf_feature = RBFSampler(gamma=1, random_state=1, n_components=40) 
        rbf_slice = rbf_feature.fit_transform(df_slice) 

        acc = classifier.score(df_slice, y_slice)
        if acc >= 0.82:
            acc_lst.append(1)
        else:
            acc_lst.append(0)
        rbf_lst.append(rbf_slice)
    return rbf_lst, acc_lst

# 40% data to train, 40% data to eval
df_train, df_eval, y_train, y_eval = train_test_split(df_rest, y_rest, test_size=0.5, random_state=12) 
rbf_lst_train, acc_lst_train = get_sample_points(df_train, y_train, classifier)
rbf_lst_eval, acc_lst_eval = get_sample_points(df_eval, y_eval, classifier)

# train RandomForest model
train_x = np.stack([np.mean(vector, axis=0) for vector in rbf_lst_train], axis=0)  # [nslice, nsample, nrbf] -> [nslice, nrbf]

# scaler is necessary here!
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
train_y = np.array(acc_lst_train)

# Initialize RandomForest model
classifier2 = RandomForestClassifier(n_estimators=100, random_state=42)
classifier2.fit(train_x, train_y)

# start eval!
# eval on train
pred_train = classifier2.predict(train_x)
true_train = train_y
cm_train = confusion_matrix(true_train, pred_train)
print("[[TN, FP],\n [FN, TP]]")
print("train")
print(cm_train)

# eval on eval
val_x = np.stack([np.mean(vector, axis=0) for vector in rbf_lst_eval], axis=0)  # [nslice, nsample, nrbf] -> [nslice, nrbf]
val_x = scaler.transform(val_x)

pred = classifier2.predict(val_x)
true = np.array(acc_lst_eval)

cm = confusion_matrix(true, pred)
print("eval")
print(cm)
