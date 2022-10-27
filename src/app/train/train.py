import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42
MIN_SAMPLE_SPLIT=4
MIN_SAMPLES_LEAF=5
N_ESTIMATORS=100
N_SPLITS = 5
PATH = './app/preprocessed_train.csv'

df = pd.read_csv(PATH)

features = ['Age', 'Sex', 'Pclass']
LABEL = 'Survived'

X = df[features]
y = df[LABEL]

kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

for train_index, test_index in kf.split(X):
    X_train, X_test = X.loc[train_index], X.loc[test_index]
    y_train, y_test = y.loc[train_index], y.loc[test_index]

    clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, bootstrap=True, criterion='entropy',
                                min_samples_leaf=MIN_SAMPLES_LEAF,
                                min_samples_split=MIN_SAMPLE_SPLIT, random_state=RANDOM_STATE)

    clf.fit(X_train, y_train)

pickle.dump(clf, open("model.pkl", "wb"))