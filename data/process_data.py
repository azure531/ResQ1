import sys
import pandas as pd
from sqlalchemy import create_engine



def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    #merge datasets
    df = messages.merge(categories,how='outer',on=['id'])
    return df

def clean_data(df):
    '''
    DESCRIPTION:
    1.split the values in 'categories' column,create a dataframe of the 36 individual category columns
    2.use the first row of categories dataframe to create column names
    3.extract a list of new column names, take everything up to the second to last character of each 
    4.rename columns of 'categories' with new column names
    '''
    categories = df['categories'].str.split(';',expand=True)
    row = categories.iloc[0]
    category_colnames=list(map(lambda x: x[:-2], row))
    categories.columns = category_colnames
    #convert category values to just 0 or 1
    #iterate through the category columns to keep only the last character of each string 
    #use map() on the column result to apply lambda function over each element of the column
    for column in categories:
        categories[column]= categories[column].map(lambda x:x[-1:])
        categories[column]= pd.to_numeric(categories[column].astype(str))
    #drop the original categories column from 'df'
    df=df.drop(['categories'],axis=1)
    #concatenate df and categories dataframe
    df = pd.concat([df,categories],axis=1)
    # drop duplicates
    df.drop_duplicates(keep='first', inplace=True)
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql(database_filename, engine, index=False,if_exists='replace')
    return df, database_filename


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