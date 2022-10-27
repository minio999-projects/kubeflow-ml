import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('./data/preprocessed_train.csv')

features = ['Age', 'Sex', 'Pclass']
LABEL = 'Survived'

X = df[features]
y = df[LABEL]

model = pickle.load(open('model.pkl', 'rb'))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state=42)

y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))