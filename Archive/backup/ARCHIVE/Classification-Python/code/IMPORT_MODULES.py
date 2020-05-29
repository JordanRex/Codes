## importing the relevant packages:

import os as os

# print/display all plots inline
import matplotlib.pyplot as plt
import seaborn as sns

# the base packages
import collections # for the Counter function
import csv # for reading/writing csv files
import pandas as pd, numpy as np, time, gc, bisect, re

# the various packages/modules used across processing (sklearn), modelling (lightgbm) and bayesian optimization (hyperopt, bayes_opt)
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing
import sklearn.decomposition as decomposition
from sklearn.cross_validation import cross_val_score, StratifiedKFold, StratifiedShuffleSplit
from sklearn.base import TransformerMixin
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sklearn.random_projection as rp
import category_encoders as ce
from sklearn.feature_selection import RFECV, VarianceThreshold

from bayes_opt import BayesianOptimization
from tqdm import tqdm
from hyperopt import hp, tpe, STATUS_OK, fmin, Trials, space_eval
from hyperopt.fmin import fmin
from hyperopt.pyll.stochastic import sample

# modelling/clustering algorithms
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans

# Evaluation of the model
from sklearn import model_selection
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from datetime import timedelta

# Exporting packages for SHAP/LIME
import shap
import lime
import lime.lime_tabular

# missing value imputation
from fancyimpute import KNN, MICE, NuclearNormMinimization

# modules to handle 'if-else' and 'for' loop breaks
from io import StringIO
from IPython import get_ipython

# pickle modules
import pickle

# modules used in the sampling class
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import AllKNN, EditedNearestNeighbours, RepeatedEditedNearestNeighbours