import pandas as pd

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    functions = [
        _actor_frequency,
        _director_frequence, 
        _bucket_contentRatings,
        _process_genres, 
        _sum_actor_facebook_likes,
        _drop, 
        _dropNaN, 
        _sort, 
        _filter, 
        _strip, 
        _fillna, 
        #_concat, 
        #join_explode, 
        #_freq
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
    processed_data = data
    for column in processed_data.columns:
        if(processed_data[column].dtype == 'object'):
            processed_data[column] = processed_data[column].fillna('unknown')
            processed_data[column] = processed_data[column].apply(lambda x: x.strip())
    if debug:
        print(f"{_strip.__name__}: Processed data shape: {processed_data.shape}")
    return processed_data


def _fillna(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
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


def _join_explode(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    processed_data = data
    processed_data = processed_data.explode(column=['actors', 'actor_facebook_likes'])
    processed_data.reset_index(drop=True, inplace=True) 
    if debug:
        print(f"{_join_explode.__name__}: Processed data shape: {processed_data.shape}")
    return processed_data

def _freq(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    processed_data = data

    frequency = processed_data['actors'].value_counts()
    processed_data['actors_Encoded'] = processed_data['actors'].map(frequency)

    frequency = processed_data['director_name'].value_counts()
    processed_data['directors_Encoded'] = processed_data['director_name'].map(frequency)
    if debug:
        print(f"{_freq.__name__}: Processed data shape: {processed_data.shape}")
    return processed_data



# TODO MOVE THIS TO A SEPARATE FILE

def _drop_names_likes(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    drop_columns = ['actor_1_facebook_likes', 
                    'actor_1_name', 
                    'actor_2_facebook_likes',
                    'actor_2_name', 
                    'actor_3_facebook_likes', 
                    'actor_3_name', 
                    'director_name', 
                    'actors', 
                    'movie_title']
    processed_data = data.drop(drop_columns, axis=1)
    if debug:
        print(f"{_drop_names_likes.__name__}: Processed data shape: {processed_data.shape}")
    return processed_data


def _genres_dummies(data: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    # TODO modify this with a lambda function
    genres = data
    genres['genres'] = genres['genres'].astype(str)
    genres['genres'] = genres['genres'].str.split('|')
    genre_dummies = genres['genres'].explode().str.get_dummies().groupby(level=0).max()
    processed_data = pd.concat([genres.drop(columns=['genres']), genre_dummies], axis=1)
    if debug:
        print(f"{_genres_dummies.__name__}: Processed data shape: {processed_data.shape}")
    return processed_data



def _bucket_contentRatings(data: pd.DataFrame) -> pd.DataFrame:
    content_rating_df = data[['content_rating']].copy()
    content_rating_df['content_rating'] = content_rating_df['content_rating'].fillna("other")
    total_count = content_rating_df['content_rating'].value_counts().sum()
    content_rating_df['percentage'] = content_rating_df['content_rating'].map(content_rating_df['content_rating'].value_counts()) / total_count * 100
    content_rating_df["rating_bin"] = content_rating_df["content_rating"].where(content_rating_df["percentage"] >= 10, "other")
    content_rating_df.drop(columns=['content_rating','percentage'], inplace=True)
    return pd.concat([data, content_rating_df], axis=1)


def _process_genres(data: pd.DataFrame) -> pd.DataFrame:
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



def _director_frequence(data: pd.DataFrame) -> pd.DataFrame:
    data['director_name'] = data['director_name'].fillna('unknown_director')
    director_frequencies = data['director_name'].value_counts()
    director_frequencies['unknown_director'] = 1 
    data['director_frequency'] = data['director_name'].map(director_frequencies)
    data = data.drop(columns=['director_name'])
    return data

def _actor_frequency(data: pd.DataFrame) -> pd.DataFrame:
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


def preprocess_data2(data: pd.DataFrame) -> pd.DataFrame:
    functions = [
        _bucket_contentRatings,
        _process_genres, 
        _director_frequence, 
        _actor_frequency
        ]
    return _pipeline(data, functions)


def _sum_actor_facebook_likes(data: pd.DataFrame) -> pd.DataFrame:
    data['actor_1_facebook_likes'] = data['actor_1_facebook_likes'].fillna(0)
    data['actor_2_facebook_likes'] = data['actor_2_facebook_likes'].fillna(0)
    data['actor_3_facebook_likes'] = data['actor_3_facebook_likes'].fillna(0)

    data['actor_total_facebook_likes'] = (
        data['actor_1_facebook_likes'] + 
        data['actor_2_facebook_likes'] + 
        data['actor_3_facebook_likes'])

    data = data.drop(columns=['actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes'])
    return data