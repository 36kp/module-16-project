import pandas as pd

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    functions = [
        _actor_frequency,
        _director_frequency, 
        _bucket_contentRatings,
        _process_genres, 
        _sum_actor_facebook_likes,
        _drop, 
        _dropNaN, 
        _sort, 
        _filter, 
        _strip, 
        _fillna
        ]
    return _pipeline(data, functions)

def _pipeline(data: pd.DataFrame, functions: list) -> pd.DataFrame:
    """
    Applies a series of transformation functions to a pandas DataFrame in sequence.

    Args:
        data (pd.DataFrame): The input DataFrame to be transformed.
        functions (list): A list of functions, where each function takes a DataFrame as input 
                          and returns a transformed DataFrame.

    Returns:
        pd.DataFrame: The transformed DataFrame after applying all the functions in the list.
    """
    for func in functions:
        data = func(data) 
    return data


def _drop(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """
    Drops specified columns from a pandas DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame from which columns will be dropped.
        debug (bool): If True, prints the function name and the shape of the processed DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with the specified columns removed.

    Notes:
        - The columns to be dropped are defined in the `drop_columns` list.
        - If a column in `drop_columns` is not present in the DataFrame, a KeyError will be raised unless handled.
    """
    drop_columns = ['movie_title',
                    'movie_imdb_link', 
                    'aspect_ratio', 
                    'plot_keywords', 
                    'color',
                    'language']
    processed_data = data.drop(drop_columns, axis=1)
    if debug:
        print(f"{_drop.__name__}: Processed data shape: {processed_data.shape}")
    return processed_data


def _dropNaN(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """
    Drops rows from a pandas DataFrame where specific columns contain NaN values.

    Args:
        data (pd.DataFrame): The input DataFrame from which rows with NaN values will be dropped.
        debug (bool): If True, prints the function name and the shape of the processed DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with rows containing NaN values in the specified columns removed.

    Notes:
        - The columns to check for NaN values are defined in the `drop_Nan_columns` list.
        - Only rows where all specified columns have non-NaN values will be retained.
    """
    drop_Nan_columns = ['title_year']
    processed_data = data.dropna(subset=drop_Nan_columns)
    if debug:
        print(f"{_dropNaN.__name__}: Processed data shape: {processed_data.shape}")
    return processed_data


def _sort(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """
    Sorts the columns of a pandas DataFrame alphabetically.

    Args:
        data (pd.DataFrame): The input DataFrame whose columns will be sorted.
        debug (bool): If True, prints the function name and the shape of the processed DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with columns rearranged in alphabetical order.

    Notes:
        - The function does not alter the order of the rows; only the column order is changed.
        - The sorted column order is determined using Python's default string sorting.
    """
    processed_data = data[sorted(data.columns)]
    if debug:
        print(f"{_sort.__name__}: Processed data shape: {processed_data.shape}")
    return processed_data


def _filter(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """
    Filters rows in a pandas DataFrame based on specified conditions and drops a specified column.

    Args:
        data (pd.DataFrame): The input DataFrame to be filtered.
        debug (bool): If True, prints the function name and the shape of the processed DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing rows that satisfy the specified conditions, 
                      with the specified column dropped.

    Conditions:
        - `gross` must be greater than 1,000.
        - `budget` must be greater than 1,000.
        - `country` must be 'USA'.
        - `title_year` must be greater than 1994.

    Notes:
        - Rows not meeting all conditions are excluded.
        - The `country` column is dropped from the resulting DataFrame.
    """
    processed_data = data[(data['gross']>1_000) & 
                          (data['budget']>1_000) & 
                          (data['country'] == 'USA') &
                          (data['title_year']>1994)].drop(columns='country')
    if debug:
        print(f"{_filter.__name__}: Processed data shape: {processed_data.shape}")
    return processed_data


def _strip(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    '''
    Strips leading and trailing whitespace from all string (object) columns in a Pandas DataFrame,
    and fills missing values in those columns with the string 'unknown'.

    Parameters:
    data : pd.DataFrame
        The input DataFrame to process. Columns with a data type of 'object' will be checked
        for missing values and whitespace.
    
    debug : bool, optional (default=False)
        If set to True, the function will print debug information, including the function name
        and the shape of the processed DataFrame.

    Returns:
    pd.DataFrame
        A DataFrame with all string (object) columns stripped of leading/trailing whitespace
        and missing values replaced by 'unknown'.
    '''
    processed_data = data
    for column in processed_data.columns:
        if(processed_data[column].dtype == 'object'):
            processed_data[column] = processed_data[column].fillna('unknown')
            processed_data[column] = processed_data[column].apply(lambda x: x.strip())
    if debug:
        print(f"{_strip.__name__}: Processed data shape: {processed_data.shape}")
    return processed_data


def _fillna(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    '''
    Replaces all missing (NaN) values in a Pandas DataFrame with -1.

    Parameters:
    data : pd.DataFrame
        The input DataFrame to process. All missing values will be replaced by -1.
    
    debug : bool, optional (default=False)
        If set to True, the function will print debug information, including the function name
        and the shape of the processed DataFrame.

    Returns:
    pd.DataFrame
        A DataFrame with all missing values replaced by -1.
    '''
    processed_data = data.fillna(-1)
    if debug:
        print(f"{_fillna.__name__}: Processed data shape: {processed_data.shape}")
    return processed_data

def _concat(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    processed_data = data
    processed_data['actors'] = processed_data[['actor_1_name', 'actor_2_name', 'actor_3_name']].agg(list, axis=1)
    processed_data['actor_facebook_likes'] = processed_data[['actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes']].agg(list, axis=1)
    if debug:
        print(f"{_concat.__name__}: Processed data shape: {processed_data.shape}")
    return processed_data



def _freq(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    '''
    Encodes the frequency of unique values in specific columns of a Pandas DataFrame.

    Specifically:
    - Encodes the frequency of values in the 'actors' column into a new column named 'actors_Encoded'.
    - Encodes the frequency of values in the 'director_name' column into a new column named 'directors_Encoded'.

    Parameters:
    data : pd.DataFrame
        The input DataFrame to process. Must include the following columns:
        - 'actors': A column containing lists of actors (or other unique identifiers).
        - 'director_name': A column containing director names (or other unique identifiers).
    
    debug : bool, optional (default=False)
        If set to True, the function will print debug information, including the function name
        and the shape of the processed DataFrame.

    Returns:
    pd.DataFrame
        A DataFrame with two new columns:
        - 'actors_Encoded': The frequency of each unique value in the 'actors' column.
        - 'directors_Encoded': The frequency of each unique value in the 'director_name' column.
    '''
    processed_data = data

    frequency = processed_data['actors'].value_counts()
    processed_data['actors_Encoded'] = processed_data['actors'].map(frequency)

    frequency = processed_data['director_name'].value_counts()
    processed_data['directors_Encoded'] = processed_data['director_name'].map(frequency)
    if debug:
        print(f"{_freq.__name__}: Processed data shape: {processed_data.shape}")
    return processed_data

#######################

def _bucket_contentRatings(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Groups content ratings in a Pandas DataFrame into buckets based on their percentage frequency.

    This function processes the 'content_rating' column by:
    - Filling missing values with "other".
    - Calculating the percentage frequency of each unique content rating.
    - Assigning content ratings with less than 10% occurrence to an "other" bucket.
    - Replacing the original 'content_rating' column with a new column named 'rating_bin', which 
      contains the binned content ratings.

    Parameters:
    data : pd.DataFrame
        The input DataFrame containing a 'content_rating' column to process.

    Returns:
    pd.DataFrame
        A modified DataFrame where:
        - The original 'content_rating' column is replaced by a new 'rating_bin' column.
        - Ratings with less than 10% occurrence are grouped into an "other" category.
    '''    
    content_rating_df = data[['content_rating']].copy()
    content_rating_df['content_rating'] = content_rating_df['content_rating'].fillna("other")
    total_count = content_rating_df['content_rating'].value_counts().sum()
    content_rating_df['percentage'] = content_rating_df['content_rating'].map(content_rating_df['content_rating'].value_counts()) / total_count * 100
    content_rating_df["rating_bin"] = content_rating_df["content_rating"].where(content_rating_df["percentage"] >= 10, "other")
    content_rating_df.drop(columns=['content_rating','percentage'], inplace=True)
    data.drop(columns=['content_rating'], inplace=True)
    return pd.concat([data, content_rating_df], axis=1)


def _process_genres(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Processes the 'genres' column in a Pandas DataFrame by splitting genres into individual binary columns
    and grouping less frequent genres into an "other_genre" category.

    This function:
    - Fills missing values in the 'genres' column with "other_genre".
    - Splits the pipe-separated genres into lists.
    - Identifies genres that appear in more than 10% of the rows as "frequent genres".
    - Creates binary columns for each frequent genre, where 1 indicates the presence of the genre in the row.
    - Creates an "other_genre" binary column to indicate the presence of infrequent genres.
    - Drops the original 'genres' column from the DataFrame.

    Parameters:
    data : pd.DataFrame
        The input DataFrame containing a 'genres' column to process. The column should contain 
        pipe-separated genre strings (e.g., "Action|Comedy|Drama").

    Returns:
    pd.DataFrame
        A modified DataFrame with:
        - Binary columns for each frequent genre.
        - An "other_genre" column for infrequent genres.
        - The original 'genres' column removed.
    '''    
    data['genres'] = data['genres'].fillna("other_genre")
    data['genres'] = data['genres'].str.split('|')
    all_genres = [genre for sublist in data['genres'] for genre in sublist]
    genre_counts = pd.Series(all_genres).value_counts()
    threshold = len(data) * 0.1
    frequent_genres = genre_counts[genre_counts > threshold].index
    for genre in frequent_genres:
        data[genre] = data['genres'].apply(lambda x: genre in x).astype(int)

    data['other_genre'] = data['genres'].apply(lambda x: any(genre not in frequent_genres for genre in x)).astype(int)
    data = data.drop(columns=['genres'])
    return data

def _director_frequency(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Processes the 'director_name' column in a Pandas DataFrame by calculating the frequency of each director
    and creating a new column with these frequency values.

    This function:
    - Fills missing values in the 'director_name' column with "unknown_director".
    - Calculates the frequency of each director based on their occurrences in the dataset.
    - Ensures "unknown_director" has a frequency of at least 1, even if not originally present.
    - Creates a new column, 'director_frequency', containing the frequency of each director.
    - Removes the original 'director_name' column from the DataFrame.

    Parameters:
    data : pd.DataFrame
        The input DataFrame containing a 'director_name' column to process.

    Returns:
    pd.DataFrame
        A modified DataFrame with:
        - A new 'director_frequency' column containing the frequency of each director.
        - The original 'director_name' column removed.
    '''    
    data['director_name'] = data['director_name'].fillna('unknown_director')
    director_frequencies = data['director_name'].value_counts()
    director_frequencies['unknown_director'] = 1 
    data['director_frequency'] = data['director_name'].map(director_frequencies)
    data = data.drop(columns=['director_name'])
    return data

def _actor_frequency(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Calculates the frequency of actors in a dataset and creates a total frequency column for each row.

    This function:
    - Fills missing values in the 'actor_1_name', 'actor_2_name', and 'actor_3_name' columns with unique placeholders.
    - Combines all actor columns into a single series to compute the frequency of each actor.
    - Maps the actor frequencies to individual columns for each actor (actor_1_frequency, actor_2_frequency, actor_3_frequency).
    - Calculates a new column, 'total_actor_frequency', as the sum of the frequencies of all three actors in each row.
    - Removes the original actor name columns and the intermediate frequency columns from the DataFrame.

    Parameters:
    data : pd.DataFrame
        The input DataFrame containing 'actor_1_name', 'actor_2_name', and 'actor_3_name' columns.

    Returns:
    pd.DataFrame
        A modified DataFrame with:
        - A 'total_actor_frequency' column representing the combined frequency of all three actors in each row.
        - The original actor name columns ('actor_1_name', 'actor_2_name', 'actor_3_name') removed.
        - The intermediate actor frequency columns ('actor_1_frequency', 'actor_2_frequency', 'actor_3_frequency') removed.
    '''    
    data['actor_1_name'] = data['actor_1_name'].fillna('unknown_actor_1_name')
    data['actor_2_name'] = data['actor_2_name'].fillna('unknown_actor_2_name')
    data['actor_3_name'] = data['actor_3_name'].fillna('unknown_actor_3_name')
    all_actors = pd.concat([data['actor_1_name'], data['actor_2_name'], data['actor_3_name']])
    actor_frequencies = all_actors.value_counts()
    data['actor_1_frequency'] = data['actor_1_name'].map(actor_frequencies)
    data['actor_2_frequency'] = data['actor_2_name'].map(actor_frequencies)
    data['actor_3_frequency'] = data['actor_3_name'].map(actor_frequencies)
    data['total_actor_frequency'] = data['actor_1_frequency'] + data['actor_2_frequency'] + data['actor_3_frequency']
    data = data.drop(columns=['actor_1_name','actor_2_name','actor_3_name'])
    data = data.drop(columns=['actor_1_frequency','actor_2_frequency','actor_3_frequency'])
    return data

def _sum_actor_facebook_likes(data: pd.DataFrame) -> pd.DataFrame:
    '''
    Calculates the total Facebook likes for all actors in each row of a dataset.

    This function:
    - Fills missing values in the 'actor_1_facebook_likes', 'actor_2_facebook_likes', and 
      'actor_3_facebook_likes' columns with 0.
    - Sums the Facebook likes across these three columns to create a new column named 
      'actor_total_facebook_likes'.
    - Removes the original columns for individual actor Facebook likes from the DataFrame.

    Parameters:
    data : pd.DataFrame
        The input DataFrame containing 'actor_1_facebook_likes', 'actor_2_facebook_likes', 
        and 'actor_3_facebook_likes' columns.

    Returns:
    pd.DataFrame
        A modified DataFrame with:
        - A new column 'actor_total_facebook_likes' containing the sum of Facebook likes for all actors.
        - The original columns 'actor_1_facebook_likes', 'actor_2_facebook_likes', and 
          'actor_3_facebook_likes' removed.
    '''
    data['actor_1_facebook_likes'] = data['actor_1_facebook_likes'].fillna(0)
    data['actor_2_facebook_likes'] = data['actor_2_facebook_likes'].fillna(0)
    data['actor_3_facebook_likes'] = data['actor_3_facebook_likes'].fillna(0)

    data['actor_total_facebook_likes'] = (
        data['actor_1_facebook_likes'] + 
        data['actor_2_facebook_likes'] + 
        data['actor_3_facebook_likes'])

    data = data.drop(columns=['actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes'])
    return data

def __remove_outliers(data: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
    '''
    Removes outliers from a specified column in a Pandas DataFrame based on the Z-score method.

    This function calculates the Z-score for each value in the specified column and removes rows where the Z-score
    exceeds a certain threshold. By default, the threshold is set to 3.0 standard deviations from the mean.

    Parameters:
    data : pd.DataFrame
        The input DataFrame containing the column with potential outliers.

    column : str
        The name of the column in the DataFrame to process for outliers.

    threshold : float, optional
        The Z-score threshold beyond which a value is considered an outlier. 
        Values with a Z-score greater than this threshold will be removed from the DataFrame.
        The default threshold is 3.0 standard deviations.

    Returns:
    pd.DataFrame
        A DataFrame with rows removed where the specified column contains outliers based on the Z-score method.
    '''
    z_scores = (data[column] - data[column].mean()) / data[column].std()
    data = data[abs(z_scores) < threshold]
    return data

def _remove_outliers_budget(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    '''
    Removes outliers from the 'budget' and 'gross' columns in a Pandas DataFrame.

    This function uses the Z-score method to identify and remove outliers from the 'budget' and 'gross' columns
    of the input DataFrame. Rows with budget or gross values that are more than 3 standard deviations from the mean
    are considered outliers and are removed.

    Parameters:
    data : pd.DataFrame
        The input DataFrame containing 'budget' and 'gross' columns to process for outliers.

    debug : bool, optional
        If set to True, the function will print debug information, including the function name
        and the shape of the processed DataFrame.

    Returns:
    pd.DataFrame
        A DataFrame with rows removed where the 'budget' or 'gross' columns contain outliers based on the Z-score method.
    '''
    processed_data = __remove_outliers(data, 'budget')
    if debug:
        print(f"{_remove_outliers_budget.__name__}: Processed data shape: {processed_data.shape}")
    return processed_data

def _remove_outliers_gross(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    '''
    Removes outliers from the 'budget' and 'gross' columns in a Pandas DataFrame.

    This function uses the Z-score method to identify and remove outliers from the 'budget' and 'gross' columns
    of the input DataFrame. Rows with budget or gross values that are more than 3 standard deviations from the mean
    are considered outliers and are removed.

    Parameters:
    data : pd.DataFrame
        The input DataFrame containing 'budget' and 'gross' columns to process for outliers.

    debug : bool, optional
        If set to True, the function will print debug information, including the function name
        and the shape of the processed DataFrame.

    Returns:
    pd.DataFrame
        A DataFrame with rows removed where the 'budget' or 'gross' columns contain outliers based on the Z-score method.
    '''
    processed_data = __remove_outliers(data, 'gross')
    if debug:
        print(f"{_remove_outliers_gross.__name__}: Processed data shape: {processed_data.shape}")
    return processed_data