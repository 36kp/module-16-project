import pandas as pd
from datetime import datetime


class PredictionDFBuilder:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        self.prediction = pd.DataFrame([{}]) 

        # Director 
        self.director_name = '' 

        # Actors
        self.actor_1_name = ''
        self.actor_2_name = ''
        self.actor_3_name = ''

        # Ratings
        self.rating = ''

        # Genres
        self.genre = 0

        self.empty()
        

    def empty(self): 
        '''
        Initializes or resets the 'prediction' DataFrame with predefined columns.

        This method ensures that the 'prediction' DataFrame contains only the specified
        columns, reindexing it to match the predefined `column_names`. If the DataFrame 
        already exists, it will be updated to include only the columns listed in `column_names`, 
        with missing columns added and existing columns not in the list dropped.
        '''

        column_names = ['Action',
                        'Adventure', 
                        'Comedy', 
                        'Crime', 
                        'Drama', 
                        'Family', 
                        'Fantasy',
                        'Horror', 
                        'Romance', 
                        'Sci-Fi', 
                        'Thriller', 
                        'actor_total_facebook_likes', 
                        'budget', 
                        'cast_total_facebook_likes', 
                        'director_facebook_likes', 
                        'director_frequency', 
                        'duration', 
                        'facenumber_in_poster', 
                        'gross',
                        'movie_facebook_likes', 
                        'num_critic_for_reviews',
                        'num_user_for_reviews', 
                        'num_voted_users', 
                        'other_genre', 
                        'rating_bin',
                        'title_year', 
                        'total_actor_frequency']
        self.prediction = pd.DataFrame(0, index=[0], columns=column_names)
        return self
    
    
    def add_actor_1(self, actor_name: str):
        self.actor_1_name = actor_name
        return self

    def add_actor_2(self, actor_name: str):
        self.actor_2_name = actor_name
        return self

    def add_actor_3(self, actor_name: str):
        self.actor_3_name = actor_name
        return self

    def add_director(self, director_name: str):
        self.director_name = director_name
        return self

    def add_rating(self, rating: str):
        self.rating = rating
        return self
    
    def add_genre(self, genre: str):
        self.genre = genre
        return self
    

    def __calculate_actor_facebook_likes(self):
        self.prediction["actor_total_facebook_likes"] = (
            self.__actor_rating(self.actor_1_name, 'actor_1_name', 'actor_1_facebook_likes')
            + self.__actor_rating(self.actor_2_name, 'actor_2_name', 'actor_2_facebook_likes') 
            + self.__actor_rating(self.actor_3_name, 'actor_3_name', 'actor_3_facebook_likes')
        )

    def __calculate_director_facebook_likes(self):
        self.prediction["director_facebook_likes"] = self.__actor_rating(self.director_name, 'director_name', 'director_facebook_likes')

    def __calculate_director_frequency(self):
        frequency = self.dataframe["director_name"].value_counts()
        self.prediction["director_frequency"] = self.dataframe['director_name'].map(frequency)


    def __actor_rating(self, name: str, name_col: str, like_col: str) -> int:
        row = self.dataframe[self.dataframe [name_col] == name]

        if not row.empty:
            return row[like_col].values[0].astype(int)
        else:
             return 0
        
    def __calculate_rating(self):
        self.prediction["rating_bin"] = self.rating

    def __calculate_genre(self):
        genres = {"Action", "Adventure", "Comedy", "Crime", "Drama", "Family",
              "Fantasy", "Horror", "Romance", "Sci-Fi", "Thriller"}
        for genre in genres:
            self.prediction[genre] = int(self.genre == genre)
        
    def __calculate_year(self):
        self.prediction["title_year"] = datetime.now().year    

    def __calculate_gross(self):
        self.prediction["gross"] = self.dataframe['gross'].mean()

    def __calculate_budget(self):
        self.prediction["budget"] = self.dataframe['budget'].mean()

    def _calculate_cast_total_facebook_likes(self):
        self.prediction["cast_total_facebook_likes"] = int(self.dataframe['cast_total_facebook_likes'].mean())

    def __calculate_duration(self):    
        self.prediction["duration"] = int(self.dataframe['duration'].mean())

    def __calculate_facenumber_in_poster(self):
        self.prediction["facenumber_in_poster"] = int(self.dataframe['facenumber_in_poster'].mean())

    def __calculate_movie_facebook_likes(self):
        self.prediction["movie_facebook_likes"] = int(self.dataframe['movie_facebook_likes'].mean())

    def __calculate_num_critic_for_reviews(self):
        self.prediction["num_critic_for_reviews"] = int(self.dataframe['num_critic_for_reviews'].mean())

    def __calculate_num_user_for_reviews(self):
        self.prediction["num_user_for_reviews"] = int(self.dataframe['num_user_for_reviews'].mean())

    def __calculate_num_voted_users(self):
        self.prediction["num_voted_users"] = int(self.dataframe['num_voted_users'].mean())    

    def __calculate_total_actor_frequency(self):
        self.prediction["total_actor_frequency"] = (
            self.dataframe["actor_1_facebook_likes"] 
            + self.dataframe["actor_1_facebook_likes"] 
            + self.dataframe["actor_3_facebook_likes"]   
        )


    def build(self) -> pd.DataFrame:
        self.__calculate_actor_facebook_likes()
        self.__calculate_director_facebook_likes()
        self.__calculate_director_frequency()
        self.__calculate_rating()
        self.__calculate_genre()

        self.__calculate_year()
        self.__calculate_gross()
        self.__calculate_budget()
        self._calculate_cast_total_facebook_likes()
        self.__calculate_duration()
        self.__calculate_facenumber_in_poster()
        self.__calculate_movie_facebook_likes()
        self.__calculate_num_critic_for_reviews()
        self.__calculate_num_user_for_reviews()
        self.__calculate_num_voted_users()
        self.__calculate_total_actor_frequency()
        return self.prediction        