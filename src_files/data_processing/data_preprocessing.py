from sklearn.preprocessing import LabelEncoder
import os
from src_files.logger import logging
import pandas as pd

def lower_casing(df):
    ##lowercasing the values:
    #list of numerical and categorical columns:
    cat_cols = df.select_dtypes(exclude='number').columns.to_list()
    cat_cols.remove('date')
    for i in cat_cols:
        if df[i].dtype != 'bool':
            df[i] = df[i].str.lower()
        else:
            df[i] = df[i].apply(lambda x:str(x).lower())
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        logging.info("Applying Label Encoder.")

        #target encoding
        le = LabelEncoder()
        df["label"] = le.fit_transform(df["label"])

        #lower casing
        logging.info('Lower Casing categorical columns.')
        df = lower_casing(df)
        logging.info('Lower Casing applied to categorical columns.')
        return df
    
    except KeyError as e:
        logging.error('Missing column in the dataframe: %s', e)
        raise

    except Exception as e:
        logging.error('Unexpected error during preprocessing: %s', e)
        raise

def main():
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logging.info('data loaded properly')

        # Transform the data
        logging.info('data preprocessing started')
        train_processed_data = preprocess_data(train_data)
        test_processed_data = preprocess_data(test_data)

        # Store the data inside data/processed
        data_path = os.path.join("./data", "processed")
        os.makedirs(data_path, exist_ok=True)
        logging.info('Saving preprocessed data to %s', data_path)
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        logging.info('Processed data saved to %s', data_path)
        
    except Exception as e:
        logging.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()