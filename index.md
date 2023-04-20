# Analysis of Covid-19 tweet Sentiment
- Yuming Chang


## Introduction/Summary
To analyze the sentiment of people's post on the social media is useful for understanding the opinion on crucial topics and providing customized service based on those information. It's impossible for people to do sentiment analysis by themselves, since the huge amount of data and bias from different person. In this project, I will introduce a method how to generate the sentiment based on the pure text using TextBlob and convert the text data into numeric matrix using TF-IDF method. After that, three machine learning and deep learning model are developed: Support Vector Machine (SVM), Convolutional Neural Network (CNN), Random Forset, Naive Bayes and XG Boost to compared the performance of the three models on the sentiment prediction. And in this project, Naive Bayes model will serve as the baseline. The confusion matrix of each model also provided to see how each model performance on different class.

### Goals
My goals will be as following bullets:

- Clean the tweet data, such as remove the special marks.

- Generate the sentiment of the tweet text using TextBlob method.

- Generate the numerical matrix using TD-IDF

- Create SVM, CNN, Random Forest, Naive Bayes and XG Boost models

- Tune Hyperparameters and Get the best models

- Compare the performance of each model and generate the confusion matrix of each model

## Results


<p align="center">
  <img src="https://github.com/changyming/8803Project/blob/webpage/CNN123.png?raw=true">
</p>


<p align="center">
  <img src="https://github.com/changyming/8803Project/blob/webpage/NB.png">
</p>


<p align="center">
  <img src="https://github.com/changyming/8803Project/blob/webpage/SVM.png">
</p>


<p align="center">
  <img src="https://github.com/changyming/8803Project/blob/webpage/RF.png">
</p>


<p align="center">
  <img src="https://github.com/changyming/8803Project/blob/webpage/XG.png">
</p>

