import pandas as pd

PATH = './app/train.csv'

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

df = pd.read_csv(PATH)
df = convert_sex(df)
df = impute_age(df)

df.to_csv('./data/preprocessed_train.csv', index=False, columns=df.columns)