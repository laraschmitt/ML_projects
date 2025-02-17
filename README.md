### Data Science/ Machine Learning Projects
Repository for small data science projects I conducted (mostly from the first weeks of the SPICED Academy Data Science Program). 
The repo includes data exploration and data wrangling in Pandas, data vizualisation and the application of different Machine Learning models (Random Forest, Linear Regression, ARIMA).

#### 1. Gapminder - Visual Data Analysis with Matplotlib & Seaborn
* dataset: [gapminder](https://www.gapminder.org/data/)
* EDA: data wrangling, descriptive statistics, exploratory plots (histogram, bar plot)
* final visualization as an animated scatterplot

<img src="01_animated_scatterplot/output.gif" alt="submission plots" width="450"/>


#### 2. Titanic dataset - Random Forest Classification
* dataset: [Titanic (Kaggle)](https://www.kaggle.com/c/titanic/data)
* EDA: descriptive statistics, exploratory plots
* bulding a baseline model & training of a logistic regression model 
* exploration of feature engineering techniques (Imputation, OneHotEncoding, Binning, Scaling)
* comparision of Decision Trees and Random Forest Classfier
* hyperparameter optimization for Random Forest Classifier

<img src="02_ML_titanic/EDA_plot_titanic.png" alt="submission plots" width="400"/>


#### 3. Bikesharing System - Regression Models

* dataset: [Capital Bike Share (Kaggle)](https://www.kaggle.com/c/bike-sharing-demand/data) 
* predicting the overall demand of bikes depending on weather conditions, day of the week
* EDA: exploring time series data in pandas and create time-related features out of timestamp, exploratory plots (time series plots) 
* feature expansion (polynomial terms, interaction terms), feature selection, regularization (Lasso)
* comparision of linear regression and gradient boosting
* hyperparameter optimization (grid search)

<img src="03_ML_regression/EDA_plot_bike_rentals.png" alt="submission plots" width="400"/>


#### 4. Lyrics CLassifier - BOW - Naive Bayes Model 
* data scraped from: [lyrics.com](lyrics.com)
* web scraping using BeautifulSoup and Regex
* tokenizing and lemmatizing using Spacy
* transform data into a TFIDF-Vectorizer
* Naive Bayes model to predict artist based on word spectrum

<img src="04_web_scraping/cloud.png" alt="submission plots" width="400"/>


#### 5. Time series analysis (Daily Temperature) - ARIMA model
* dataset: historical temperature data for Berlin-Tempelhof from [https://www.ecad.eu/](https://www.ecad.eu/)
* exploring Linear Autoregression "manually"
* ARIMA model to predict the daily mean temperature based on historical weather data
* examination of (partial) autocorrelations

<img src="05_time_series/model_plot_tempelhof.png" alt="submission plots" width="400"/>

