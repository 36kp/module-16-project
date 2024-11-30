import pandas as pd

def encode_data(data: pd.DataFrame):
    """Encode the rating_bin column using LabelEncoder.
    Args:
        data (pd.DataFrame): DataFrame containing the rating_bin column.

    Returns:
        pd.DataFrame: Category encoded DataFrame.
    """
    # LabelEncode rating_bin column
    data['rating_bin'] = data['rating_bin'].astype('category')
    data['rating_bin'] = data['rating_bin'].cat.codes
    return data
