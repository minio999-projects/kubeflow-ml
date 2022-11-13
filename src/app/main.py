import kfp
from kfp import components
from kfp.v2 import compiler, dsl
from kfp.v2.dsl import Input, InputPath, Model, Output, OutputPath, component
from pyexpat import model


@component(packages_to_install=['sklearn', 'pandas', 'numpy', 'pyarrow', 'fastparquet', 'scikit-learn'])
def load_data(output_file: OutputPath('parquet')):
    import numpy as np
    import pandas as pd
    from sklearn.datasets import load_wine
    
    data = load_wine()
    data=pd.DataFrame(data=np.c_[data['data'],data['target']],columns=data['feature_names']+['target'])
    data.to_parquet(output_file)

@component(packages_to_install=['sklearn', 'pandas', 'numpy', 'pyarrow', 'fastparquet', 'scikit-learn'])
def model_selection(file_path: InputPath('parquet')):
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import (BaggingClassifier, ExtraTreesClassifier,
                                  RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import KFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC, LinearSVC, NuSVC

    data = pd.read_parquet(file_path)
    features = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids',
                'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines',
                'proline']
    X = data[features]
    y = data['target']

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
    
    for model in models:
        print(model)
        scores=[]
        for train_index, test_index in kf.split(X):
            X_train, X_test = X.loc[train_index], X.loc[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]

            clf = model

            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)

            acc_score = round(accuracy_score(y_test, y_predict),3)

            print(acc_score)

            scores.append(acc_score)

        print()
        print("Average:", round(100*np.mean(scores), 3), "%")
        print("Std:", round(100*np.std(scores), 3), "%")
        print()

@component(packages_to_install=['sklearn', 'pandas', 'numpy', 'pyarrow', 'fastparquet', 'scikit-learn'])
def hyperparameter_tuning(file_path: InputPath('parquet')):
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, train_test_split

    data = pd.read_parquet(file_path)
    features = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids',
                'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines',
                'proline']
    X = data[features]
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier()

    grid_values = {
        'n_estimators': [10, 100, 1000],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]
    }

    grid = GridSearchCV(estimator = clf, param_grid = grid_values, scoring = 'accuracy',
                        cv = 3, refit = True, return_train_score = True)
    grid.fit(X_train, y_train)
    print(grid.best_estimator_)
    print(grid.best_score_)
    print(grid.best_params_)

@component(packages_to_install=['pandas', 'numpy', 'pyarrow', 'sklearn', 'scikit-learn'])
def train(file_path: InputPath('parquet'), clf: Output[Model]):
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import KFold

    data = pd.read_parquet(file_path)
    features = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids',
                'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines',
                'proline']
    X = data[features]
    y = data['target']

    kf = KFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]

        clf = RandomForestClassifier(n_estimators=1000)

        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)

        acc_score = round(accuracy_score(y_test, y_predict),3)

        print(acc_score)

        scores.append(acc_score)

    print("Average:", round(100*np.mean(scores), 3), "%")
    print("Std:", round(100*np.std(scores), 3), "%")

@dsl.component
def test(clf: Input[Model]):
    print(clf)
@dsl.pipeline(name='test')
def pipeline():
    data = load_data()
    model_selection(data.output)
    hyperparameter_tuning(data.output)
    step = train(data.output)
    test(step.output)
if __name__ == '__main__':
    kfp.compiler.Compiler(mode=kfp.dsl.PipelineExecutionMode.V2_COMPATIBLE).compile(
    pipeline_func=pipeline,
    package_path='pipeline.yaml')
