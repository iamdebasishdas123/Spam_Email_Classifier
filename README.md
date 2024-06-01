### Spam Email Classifier Streamlit Link:scamemailcassifier-email
.streamlit.app

In this project, we aim to build a machine-learning model that can accurately distinguish between spam and ham (non-spam) messages. This comprehensive journey includes data cleaning, exploratory data analysis (EDA), text preprocessing, model building, and improvement. Finally, we'll discuss deployment to a website.

## 1. Data Cleaning

Our dataset consists of 5,572 entries and five columns, with three of these columns containing mostly null values. We begin by dropping these unnecessary columns and renaming the remaining ones for clarity.

After these adjustments, we have a dataset with the following structure:
- **target**: Indicates whether the message is spam (1) or ham (0).
- **text**: The content of the message.

Next, we encode the target labels from text to numeric values using a Label Encoder and remove any duplicate entries, reducing our dataset to 5,169 unique messages.

## 2. Exploratory Data Analysis (EDA)

To understand our dataset better, we examine the distribution of spam and ham messages. The dataset is imbalanced, with a higher proportion of ham messages compared to spam. Visualization using a pie chart illustrates this imbalance clearly.

Further, we generate descriptive statistics for the length of messages, the number of words, and the number of sentences in both spam and ham categories. Spam messages tend to be longer and contain more words and sentences on average compared to ham messages.

Pairplots and heatmaps help visualize the relationships and correlations between these features, giving us insights into the data patterns.

## 3. Data Preprocessing

To prepare the text data for model building, we perform several preprocessing steps:
- **Lowercasing**: Converting all text to lowercase to maintain consistency.
- **Tokenization**: Splitting the text into individual words.
- **Removing Special Characters**: Keeping only alphanumeric characters.
- **Removing Stop Words and Punctuation**: Filtering out common but non-informative words.
- **Stemming**: Reducing words to their root forms.

Using the transformed text, we create new columns in the dataset that reflect these changes.

## 4. Text Visualization

Word clouds and bar plots are used to visualize the most common words in spam and ham messages. These visualizations help us understand the predominant words and terms used in each category, further aiding in feature selection for our models.

## 5. Model Building

We explore three different models for our task: Gaussian Naive Bayes, Multinomial Naive Bayes, and Bernoulli Naive Bayes. Using the TfidfVectorizer, we convert the text data into a matrix of TF-IDF features.

After splitting the data into training and testing sets, we train and evaluate each model:
- **Gaussian Naive Bayes**: Achieves an accuracy of around 86.94%, but with a lower precision score for spam detection.
- **Multinomial Naive Bayes**: Performs the best with a high accuracy of 97.10% and perfect precision.
- **Bernoulli Naive Bayes**: Also performs well with an accuracy of 96.95%.

The Multinomial Naive Bayes model emerges as the best performer, with the highest accuracy and precision, making it our model of choice for deployment.

## 6. Model Improvement and Hyperparameter Tuning

To further enhance our model's performance, we can:
- **Optimize Hyperparameters**: Using techniques like GridSearchCV to find the best parameters for our model.
- **Feature Selection**: Experimenting with different features and n-grams to improve model accuracy.
- **Balancing the Dataset**: Employing techniques like SMOTE to address the data imbalance.

## 7. Deployment

Finally, deploying the model involves creating a user-friendly interface, such as a web application, where users can input text messages and get predictions on whether they are spam or ham. Tools like Flask or Django can be used to build this interface, and platforms like Heroku or AWS can host the application.

## Conclusion

Building a spam detection model is a multi-step process that involves cleaning and preprocessing the data, exploring various models, and optimizing their performance. The final model, once deployed, can effectively classify messages as spam or ham, providing a practical solution to a common problem in digital communication. This project highlights the importance of each step in the machine learning pipeline and the value of continuous improvement and testing.
