import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load data from files and return a combined DataFrame.
    Args:
        messages_filepath (str): The file path for the messages data.
        categories_filepath (str): The file path for the categories data.
    Returns:
        pandas.DataFrame: A combined DataFrame containing the loaded data.
    """
    # load the 2 datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge the datasets
    df = pd.merge(messages, categories, on='id', how='inner')
    return df

def clean_data(df_total):
    """
    clean the total dataset
    Args:
        df_total (pandas.DataFrame): The input dataframe.
    Returns:
        pandas.DataFrame: The cleaned dataframe.
    """
    # create a dataframe of the 36 individual category columns
    categories = df_total['categories'].str.split(';', expand=True)
    categories.columns = categories.iloc[0].apply(lambda x: x.split('-')[0])
    # retrieve the values of each column
    categories = categories.apply(lambda x: x.str.split('-').str[1])
    # convert column from string to numeric
    for column in categories:
        categories[column] = categories[column].astype(int)
    # for column "related", convert the 2 -> 1
    categories.loc[categories['related'] == 2,'related'] = 1
    # drop the original categories column from `df`
    df_total.drop('categories', axis=1, inplace=True)
    df_total = pd.concat([df_total, categories], axis=1)
    # drop duplicates
    df_clean = df_total.drop_duplicates()
    return df_clean

def save_data(df_clean, database_filename):
    """
    save the cleaned dataframe into the sql database
    Args:
        df_clean (pandas.DataFrame): The cleaned dataframe.
        database_filename (str): the file name of the sql database
    Returns:
        None.
    """
    engine = create_engine('sqlite:///' + database_filename)
    df_clean.to_sql('resp_messages', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()