import sys
import pandas as pd
from sqlalchemy import create_engine


def slicing(x):
     
        """Extract only the category:
        - splits categories into separated columns
        - converts categories values to binary values
        - drop duplicated rows
    
    Args:
        category-X (string -> X = binary number) : category raw format
    Returns:
        category (string): Cleaned category without binary number
    """    
    
    return x[:-2]

def load_data(messages_filepath, categories_filepath):
    
    """
    Load the data from message and categories files to a dataframe
    
    Args:
        messages_filepath (string): The file path of messages file
        categories_filepath (string): The file path of categories file
    Returns:
        df (dataframe): Merged messages and categories df by ID.
    """
    
   
    messages_filepath = pd.read_csv(messages_filepath)
    categories_filepath = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages_filepath.merge(categories_filepath, on = 'id')
    
    return df       



def clean_data(df):
    
    """Clean the data:
        - splits categories into separated columns
        - converts categories values to binary values
        - drop duplicated rows
    
    Args:
        df (pandas dataframe): merged categories and messages dataframes
    Returns:
        df (pandas dataframe): Cleaned dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';')
    
    # select the first row of the categories dataframe
    category = categories[0]
    category_colnames = []

    for i in category:
        category_colnames.append(slicing(i))
    
    # rename the columns of categories`
    df = df.reindex(columns = df.columns.tolist() + category_colnames)
    
    #Convert category values to just numbers 0 or 1

    cont = 0
    list_cat = []
    list_cat_num = []
    final_list = []

  #Function to get the list with category numbers for each message/line
    for line in df['categories']:
        list_cat = line.split(';')
        for i in list_cat:
            list_cat_num.append(int(i[-1]))
        final_list.append(list_cat_num)
        list_cat_num = []
        list_cat = []

    #concatenate the original dataframe with the new `categories` dataframe  
    df_category = pd.DataFrame(final_list, columns = category_colnames)
    df_first_part = df.loc[:,['id','message','original','genre']]
    
    concatened = pd.concat([df_first_part,df_category ],sort=False, axis = 1)
       
    # remove the duplicated rows
    
    concatened = concatened.drop_duplicates(keep = 'first')

    # Some numbers are non binary (2). Let's replace them to 1
    concatened['related'] = concatened['related'].replace(2,1)
    
    return concatened 


def save_data(df, database_filename):
    
    """Save the processed data to a sqlite db
    Args:
        df (pandas dataframe): The processed dataframe
        database_filename (string): the file path 
    Returns:
        None
    """
    
    #Save the clean dataset into an sqlite database
    table_name = 'Messages'   

    engine = create_engine('sqlite:///{}'.format(database_filename))
    
    df.to_sql(table_name, engine, index=False,if_exists='replace')
    

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
