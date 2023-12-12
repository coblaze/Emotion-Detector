# Emotion-Detector
This repository contains a Python project that performs sentiment analysis and emotion detection on text data. It uses the GoEmotions dataset by Google for training, and includes code to connect to Twitter API for real-time analysis. The results are visualized using pie charts. For more information about the process, visit cosei.io.


## How do words feel? Exploring Sentiment Analysis and Emotion Detection

In this project, we delve into the fascinating world of sentiment analysis and emotion detection. We tackle essential tasks like text preprocessing and feature engineering, exploring a variety of machine learning techniques to create models that can classify and evaluate text data. We assess our models using a tool called a confusion matrix.

We use sentiment analysis to assess the overall sentiment of a sentence, categorizing it as positive, negative, or neutral. This offers insights into user reactions to products or brands. However, this technique has some limitations such as its inability to capture the full spectrum of emotions. To overcome this limitation, we use emotion detection.

Emotion detection identifies specific emotions like sadness, anger, and happiness in text data. This offers businesses a more comprehensive understanding which makes facilitating informed decision-making easier.

While there are many libraries available for predicting sentiments in text, the same doesn't hold true for detecting emotions which is a bit more complex. To handle this problem we create a custom classifier. This classifier classifies emotions alongside the sentiment prediction libraries to assess both the emotional and sentiment aspects of text.

For more information about the process visit [cosei.io](http://cosei.io)

### About the Dataset

The GoEmotions dataset comprises 58,000 meticulously selected Reddit comments annotated across 27 distinct emotion categories alongside a neutral classification. These categories span a comprehensive spectrum of human emotional responses, encompassing complex nuances such as admiration, amusement, anger, and more. Each comment serves as a valuable data point contributing to a profound understanding of how individuals express a diverse range of emotions within online communities.

This dataset stands as a robust resource for academic and professional endeavors offering rich insights into the intricate tapestry of human emotional experiences in digital communication.

For access to the dataset please follow this link: https://github.com/google-research/google-research/blob/master/goemotions/README.md

### How to Run the Code

To run the code, you need to have Python installed on your machine. You also need to install several Python libraries including pandas, nltk, textblob, sklearn, xgboost, chart_studio and plotly. You can install these using pip:

```bash
pip install pandas nltk textblob sklearn xgboost chart_studio plotly
```

Once you have these installed, you can run the code in a Python environment or Jupyter notebook.

### Output

The output of the code is a sentiment analysis and emotion detection of the given text data. The code also connects to Twitter API and performs sentiment analysis and emotion detection on tweets related to a specific topic. The results are visualized using pie charts.

For any further questions or inquiries about this project please visit [cosei.io](http://cosei.io)
