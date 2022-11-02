from pyexpat import model
import kfp
from kfp import components
from kfp.v2 import compiler, dsl
from kfp.v2.dsl import component, Input, Output, InputPath, Dataset, Model

@component(packages_to_install=['sklearn'])
def load_data() -> Dataset:
    from sklearn.datasets import load_wine
    
    data = load_wine()
    return data

@component(packages_to_install=['sklearn', 'pandas', 'numpy'])
def model_selection(data) -> list:
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import KFold
    from sklearn.svm import LinearSVC, NuSVC, SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
    from sklearn.metrics import accuracy_score

    X = data.data
    y = data.target

    models = [
        SVC(), NuSVC(), LinearSVC(),
        SGDClassifier(), KNeighborsClassifier(),
        LogisticRegression(),BaggingClassifier(),
        ExtraTreesClassifier(), RandomForestClassifier()
    ]
    
    kf = KFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )
    
    scores = []
    best_model = ['dummy', 0]
    
    for model in models:
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]

            clf = model

            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)

            acc_score = round(accuracy_score(y_test, y_predict),3)

            print(acc_score)

            scores.append(acc_score)
            
            average = 100*np.mean(scores) 
            std = 100*np.std(scores)
            if std < 3 and average > best_model[1]:
                best_model.clear()
                best_model.append(str(model))
                best_model.append(average)
                best_model.append(std)

        print()
        print(model)
        print("Average:", round(100*np.mean(scores), 1), "%")
        print("Std:", round(100*np.std(scores), 1), "%")
    return best_model

@dsl.pipeline(name='test')
def pipeline():
    model_selection(load_data().output)

if __name__ == '__main__':
    kfp.compiler.Compiler(mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE).compile(
    pipeline_func=pipeline,
    package_path='pipeline.yaml')