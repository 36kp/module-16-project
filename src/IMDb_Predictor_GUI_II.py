import tkinter as tk
from tkinter import ttk
import utils.fetcher_utils as fetcher
import pandas as pd
from utils.prediction_builder import PredictionDFBuilder 

model = ""
df = pd.DataFrame()

class PredictionDFBuilderGUI:
    
    def __init__(self, master, df, model):
        self.master = master
        self.model = model
        self.df = df
        master.title("PredictionDF Builder")
        
        # Variables for dropdown selections
        self.actor_1 = tk.StringVar()
        self.actor_2 = tk.StringVar()
        self.actor_3 = tk.StringVar()
        self.director = tk.StringVar()
        self.rating = tk.StringVar()
        self.genre = tk.StringVar()

        #List of options for dropdowns without 'Select Another'
        # self.actors = ["Orlando Bloom", "Meryl Streep", "Tom Hanks"]
        self.directors = ["Steven Spielberg", "Gore Verbinski"]
        self.ratings = ["PG", "PG-13", "R", "G", "NC-17"]
        self.genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Horror", "Romance", "Fantasy", "Mystery", "Thriller"]

        df['actor_1_name'] = df['actor_1_name'].dropna()
        self.actors_1 = df['actor_1_name'].tolist()
        df['actor_2_name'] = df['actor_2_name'].dropna()
        self.actors_2 = df['actor_2_name'].tolist()
        df['actor_3_name'] = df['actor_3_name'].dropna()
        self.actors_3 = df['actor_3_name'].tolist()
        # self.directors = df['director_name'].unique()

        # Labels and Dropdowns
        tk.Label(master, text="Actor 1:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        ttk.Combobox(master, textvariable=self.actor_1, values=self.actors_1).grid(row=0, column=1, padx=5, pady=5)

        tk.Label(master, text="Actor 2:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        ttk.Combobox(master, textvariable=self.actor_2, values=self.actors_2).grid(row=1, column=1, padx=5, pady=5)

        tk.Label(master, text="Actor 3:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        ttk.Combobox(master, textvariable=self.actor_3, values=self.actors_3).grid(row=2, column=1, padx=5, pady=5)

        tk.Label(master, text="Director:").grid(row=3, column=0, sticky="e", padx=5, pady=5)
        ttk.Combobox(master, textvariable=self.director, values=self.directors).grid(row=3, column=1, padx=5, pady=5)

        tk.Label(master, text="Rating:").grid(row=4, column=0, sticky="e", padx=5, pady=5)
        ttk.Combobox(master, textvariable=self.rating, values=self.ratings).grid(row=4, column=1, padx=5, pady=5)

        tk.Label(master, text="Genre:").grid(row=5, column=0, sticky="e", padx=5, pady=5)
        ttk.Combobox(master, textvariable=self.genre, values=self.genres).grid(row=5, column=1, padx=5, pady=5)

        # Build Button
        ttk.Button(master, text="Build Prediction DF", command=self.build_prediction_df).grid(row=6, column=1, pady=10)

        # Output Label
        self.output_label = tk.Label(master, text="")
        self.output_label.grid(row=7, column=0, columnspan=2, pady=10)

    def build_prediction_df(self):
        # Assuming you have these classes/functions defined elsewhere
        
        builder = PredictionDFBuilder(df)
        prediction_df = (
            builder
            .add_actor_1(self.actor_1.get())
            .add_actor_2(self.actor_2.get())
            .add_actor_3(self.actor_3.get())
            .add_director(self.director.get())
            .add_rating(self.rating.get())
            .add_genre(self.genre.get())
            .build()
        )
        
        prediction = model.predict(prediction_df)
        self.output_label.config(text=f"Predicted IMDb Score: {prediction[0]:.2f}")

# if __name__ == "__main__":
#     root = tk.Tk()
#     gui = PredictionDFBuilderGUI(root)
#     root.mainloop()