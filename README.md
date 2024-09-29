# Twitter Sentiment Analysis

This project focuses on sentiment analysis of Twitter data using a dataset from Kaggle. It preprocesses the data and builds a classification model to determine whether a tweet has a positive, negative, or neutral sentiment.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project uses a dataset sourced from Kaggle to analyze the sentiment of tweets. The process includes:

- Installing necessary dependencies.
- Fetching and extracting the dataset.
- Preprocessing the text data by removing stop words and performing stemming.
- Building a classification model using Logistic Regression to classify tweets as positive, negative, or neutral.

## Installation

Follow these steps to set up the project:

1. **Install Kaggle Library:**
   The project uses the Kaggle API to download the dataset:
   ```bash
   pip install kaggle
   ```

2. **Configure Kaggle Credentials:**
   Ensure you have a Kaggle API key (`kaggle.json`) configured:
   ```bash
   # Copy kaggle.json to the appropriate directory
   mkdir -p ~/.kaggle
   cp /path/to/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. **Download the Dataset:**
   Use the Kaggle API to download the sentiment analysis dataset:
   ```bash
   kaggle datasets download -d kazanova/sentiment140
   ```

4. **Extract the Dataset:**
   Extract the dataset from the zip file:
   ```python
   from zipfile import ZipFile
   dataset = '/content/sentiment140.zip'
   with ZipFile(dataset, 'r') as zip:
       zip.extractall()
       print("The dataset is extracted")
   ```

## Usage

1. **Preprocess the Data:**
   The text data is cleaned and preprocessed, including the removal of stopwords and stemming using `PorterStemmer` from the NLTK library.
   
   Example of importing dependencies:
   ```python
   import re
   import nltk
   from nltk.corpus import stopwords
   from nltk.stem.porter import PorterStemmer
   from sklearn.feature_extraction.text import TfidfVectorizer
   ```

2. **Load the Dataset:**
   Load the sentiment dataset into a Pandas DataFrame:
   ```python
   twitter_data = pd.read_csv('/content/training.1600000.processed.noemoticon.csv', encoding='ISO-8859-1')
   ```

3. **Train the Model:**
   Split the data into training and testing sets, then use Logistic Regression to train the sentiment classifier:
   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.linear_model import LogisticRegression
   from sklearn.metrics import accuracy_score

   model = LogisticRegression()
   model.fit(X_train, y_train)
   ```

## Technologies Used

- **Python**
- **Libraries:**
  - Numpy
  - Pandas
  - Scikit-learn
  - NLTK (Natural Language Toolkit)

## Dataset

The dataset used for this project is the **Sentiment140 dataset** from Kaggle. It contains 1.6 million tweets labeled with sentiment (positive, negative, or neutral).

- **Source:** [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)

## Results

The Logistic Regression model was trained on the dataset and evaluated using accuracy. Further results are available in the notebook.

## Contributing

If you'd like to contribute, feel free to submit a pull request or raise an issue.

## License

This project is licensed under the MIT License.


