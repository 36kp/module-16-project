# CinemaScore Predictor: An AI Model for IMDb Rating Prediction

## Table of Contents
- [Goals](#goals)
  - [Model Development](#model-development)
  - [Performance Metrics](#performance-metrics)
- [Deployment](#deployment)
- [Future Enhancements](#future-enhancements)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
---
## Purpose
The purpose of this project is to develop a supervised machine learning model that can predict the IMDb score of a movie based on key attributes such as the director, budget, lead actor, and genre. This tool aims to assist film studios, investors, and enthusiasts in predicting the potential reception of a movie before its release, thereby aiding in strategic decision-making for production, marketing, and investment.

---

## Dataset
### Source:
We will compile data from the IMDb movie database. This dataset will include:
- **Director:** Names of directors.
- **Budget:** The total budget in USD.
- **Actor:** Names of lead actors.
- **Genre:** Categorized genres of the movie.
- **IMDb Score:** The actual IMDb score of the movie.

### Size:
- **5,000 movies** to ensure the model has enough data to learn from, considering the variety in directors, actors, and genres.

### Cleaning:
- Handle missing values.
- Normalize budget figures.
- Encode categorical variables (e.g., director, actor, genre).

---

## Initial Analysis
### Exploratory Data Analysis (EDA):
- Visualize the distribution of IMDb scores.
- Analyze the correlation between budget and IMDb scores.
- Explore how different genres affect scores.

### Feature Engineering:
- Create interaction features between director-genre, actor-genre, etc., to capture combined effects.
- Use techniques like one-hot encoding or embedding for categorical variables.

### Data Insights:
- Identify patterns or anomalies, such as whether certain directors consistently perform better in specific genres or if high-budget films correlate with higher or lower scores.

---

## Goals
### Model Development:
- **Regression Models:** Develop and evaluate regression models for training.
- **Classification Models:** Develop and evaluate classification models for training.

### Performance Metrics:
- Use **Mean Squared Error (MSE)**, **Mean Absolute Error (MAE)**, and **R-squared** to evaluate model performance.
- Implement **cross-validation** to ensure model robustness.

---

## Deployment
- Develop a simple **web interface** or an **API** where users can input movie details and receive a predicted IMDb score.
- Ensure the model can be updated periodically with new movie data to keep predictions relevant.

---

## Future Enhancements
- Integrate more variables like **release date**, **production company**, or sentiment analysis from movie trailers or reviews to improve prediction accuracy.
- Consider **real-time updates** or predictions based on initial audience reactions or festival screenings.
- Integrate other databases, including **Box Office Mojo** and **The Numbers**, for better predictive analysis.

---

This proposal outlines a structured approach to developing a predictive model for movie success, potentially revolutionizing how films are evaluated before they hit the screens.

---
## Project Structure
```
module-16-project/ 
    ├── docs/ 
    │   ├── Final-Presentation.pdf              # Final project presentation
    │   ├── module-16-project.pptx              # Powerpoint file for the presentation
    │   └── proposal.md                         # Project proposal
    ├── resources/
    │   ├── img/                                # Project images and exported charts
    │   ├── movie_metadata.csv                  # Downloaded CSV file (Available only after first run)
    |   └── preprocessed_df.csv                 # Processed and cleaned up dataframe exported (Available only after first run)
    ├── src/                                    # Source code files
    |   └── analysis/                           # Jupyter notebooks used for data analysis
    |       ├── analysis_*.ipynb                # Detailed analysis file for each features
    |       └── prediction_test.ipynb           # Pipeline and model prediction test file
    ├── utils/                                  # Utility files (Python modules)
    |   ├── fetcher_utils.py                    # Data fetching code
    |   ├── pipeline_util.py                    # Model training pipeline code
    |   ├── prediction_builder.py               # Prediction DataFrame builder code
    |   ├── preprocess_util.py                  # Data cleanup and aggregation functions
    |   ├── trainer_util.py                     # Correlation analysis and model training implementation
    |   └── transformer_util.py                 # Data encoding functions
    ├── IMDb_Predictor_GUI_II.py                # GUI for Data Prediction application
    ├── main.ipynb                              # Main project entrypoint notebook
    └── model_performance_analysis.ipynb        # Model performance analysis notebook

```
## Prerequisites
- Python 3.7 or higher
- Jupyter Notebook
- Required Python packages:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn

You can install the required packages using the following command:
```sh
pip install pandas numpy scikit-learn matplotlib seaborn
```
## Usage

1. Clone the repository
```sh
git clone https://github.com/36kp/module-16-project.git
cd module-16-pproject
```
2. Open Jupyter Notebook:
```sh
jupyter notebook
```
3. Navigate to the `src` folter and open and execute `main.ipynb`
4. For more detailed analysis, explore the notebooks in the `src/analysis` folder
---
## Contributing
We welcome contributions to the project! If you'd like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request describing your changes.
---
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.