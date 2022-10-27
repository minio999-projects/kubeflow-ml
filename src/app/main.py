import kfp
from kfp import dsl
from kfp.components import func_to_container_op, InputPath, OutputPath
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

PATH = './data/train.csv'

@func_to_container_op
def get_df(path: InputPath('CSV')) -> pd.DataFrame:
    """ Function generating pandas dataframe

    Args:
        path (str): path to train file

    Returns:
        pd.dataframe: returns dataframe on which we will oparate
    """

    df = pd.read_csv(path)
    print(df.head(5))
    return df

@func_to_container_op
def impute_age(df: pd.DataFrame) -> pd.DataFrame:
    """ Function which imputes mean age to null values in age col

    Args:
        df (pd.DataFrame): dataframe on which we will oparate

    Returns:
        pd.DataFrame: imputed dataframe 
    """
    mean_age = df['Age'].mean()
    df['Age'] = df["Age"].fillna(mean_age)
    return df

@func_to_container_op
def convert_sex(df: pd.DataFrame) -> pd.DataFrame:
    """converting sex columns from 'male' or 'female' into 1 or 0

    Args:
        df (pd.DataFrame): df on which to oparate

    Returns:
        pd.DataFrame: converted dataframe
    """
    df['is_male'] = 0
    df.loc[df['Sex'] == 'male', 'is_male'] = 1
    df = df.drop(columns=['Sex'])
    return df

if __name__ == '__main__':
    # Compiling the pipeline
    kfp.compiler.Compiler().compile(flipcoin_exit_pipeline, __file__ + '.yaml')