# 🌍 Disaster Tweet Prediction 🌪️

## 📖 Introduction
In this project, we aim to predict whether a tweet is about a disaster or not. This classification task involves analyzing tweet text data to determine if the tweet refers to a natural disaster, such as an earthquake, fire, or flood. The goal is to create a model that can accurately classify tweets into two categories:

- **Disaster**: The tweet is related to a natural disaster or emergency.
- **Not Disaster**: The tweet is not related to a natural disaster, and could refer to any other topic.

The dataset contains tweets with various types of content, and the goal is to identify patterns in the text data that indicate whether the tweet is about a disaster.

## 📝 Dataset Columns

- **id**: Unique identifier for each tweet.
- **keyword**: A keyword from the tweet, which might be related to the disaster. This column could provide additional context but is not always reliable, as some keywords may be missing or generic.
- **location**: The location associated with the tweet. This column may contain geolocation information or could be empty if no location was provided by the user.
- **text**: The text content of the tweet. This is the main feature we will be analyzing to determine if the tweet is about a disaster.
- **target**: The target label indicating if the tweet is about a disaster (1) or not (0). This column will be used as the ground truth in the model's training and evaluation.

## 🎯 Objective
Our objective is to build a machine learning model that predicts the value of the **target** column based on the **text** data, effectively classifying tweets as related to disasters or not. The model will be trained using features extracted from the tweet text and evaluated for its accuracy in making predictions.

## 📊 Model Performance

Here are the accuracy scores of the various models tested:

| Model                       | Accuracy Score |
|-----------------------------|----------------|
| **Logistic Regression**      | 0.796454       |
| **Random Forest Classifier** | 0.780696       |
| **Bernoulli Naive Bayes**    | 0.772817       |
| **Multinomial Naive Bayes**  | 0.772817       |
| **Gradient Boosting**        | 0.751806       |
| **Decision Tree Classifier** | 0.741300       |
| **AdaBoost Classifier**      | 0.741300       |

## 📈 Key Insights
- **Logistic Regression** performed the best with an accuracy score of **0.796454**, making it the most reliable model for this task.
- **Random Forest Classifier** came close with an accuracy of **0.780696**.
- **Bernoulli Naive Bayes** and **Multinomial Naive Bayes** achieved accuracy scores of **0.772817**.
- **Gradient Boosting**, **Decision Tree**, and **AdaBoost Classifiers** showed slightly lower performance with accuracy ranging between **0.741300** and **0.751806**.

## 🏁 Conclusion
Overall, **Logistic Regression** appears to be the most effective model for classifying disaster-related tweets in this dataset. While other models like **Random Forest** and **Naive Bayes** also showed promising results, **Logistic Regression** should be considered the preferred model due to its higher accuracy.

## 🔗 Useful Links

- [Kaggle Notebook](https://www.kaggle.com/code/senasudemir/disaster-tweet-prediction?scriptVersionId=222821208)
- [Hugging Face Space](https://huggingface.co/spaces/Senasu/Disaster_Tweet_Prediction)
- [Dataset Link](https://www.kaggle.com/competitions/nlp-getting-started/data)
