# Executive Summary
________

This data science project aimed to create the highest performing model that can correctly classify individual posts to either Showerthoughts and CasualConversation. Leveraging machine learning algorithms on historical subreddit data from both Showerthoughts and CasualConversation communities, this project produced four models that the social media startup can use and modify to build their personality app that classifies people to either happy or sad based on text data. As the best performing model out of the four, the Logistic Regression model achieved an 79 percent accuracy rate in distinguishing between Showerthoughts and CasualConversation, saving the company time and resources in creating the backbone of its personality app. 


## Data Collection and Description  
________

- PRAW method for collecting subreddit data

- Data Dictionary

| Variable     | Data Type  | Value Count  | Description                                   |
| -----------  | --------   | ------------ | ------------------------                      |
| title        | object     | 2365         | title of individual post in subreddit         |
| selftext     | object     | 1370         | text within individual post in subreddit      | 
| subreddit    | int64      | 2365         | 1 - Showerthoughts and 0 - CasualConversation | 
| created_utc  | float64    | 2365         | unix timestamp                                | 
| name         | object     | 2365         | username of individual in Reddit              | 

- Data Visualizations

![Top Ten Common Words from Subreddit](https://git.generalassemb.ly/daniel613/project_3/blob/main/images/Top%2010%20Common%20Words%20in%20Title%20from%20Subreddit.png)

![CM of Bernoulli with Hypertuned CountVectorizer](https://git.generalassemb.ly/daniel613/project_3/blob/main/images/Confusion%20Matrix%20of%20Bernoulli%20with%20Hypertuned%20CountVectorizer.png)

## Model Preprocessing Procedure
_________

- Supervised Machine Learning
- Natural Language Processing

***cvec stands for CountVectorizer
***tvec stands for TfidVectorizer

- Target: subreddit
- Feature: title
- CountVectorizer and TfidVectorizer Transformers
- Models: Logistic Regression, Bernoulli, Multinomial, Random Forest

- train_test_split: test_size=0.25, random_state=42, stratify=y 

- 'cvec__max_features' : [1500, 3000, 5000],
- 'cvec__min_df' : [0.03, 0.05],
- 'cvec__max_df': [0.85, 0.9],
- 'cvec__ngram_range' : [(1, 1), (1, 2)] 

- 'tvec__max_features' : [1500, 3000, 5000],
- 'tvec__min_df' : [0.03, 0.05],
- 'tvec__max_df': [0.85, 0.9],
- 'tvec__ngram_range' : [(1, 1), (1, 2)]

- Set GridSearchCV
- Set cross validation to 5

***second iteration only

- custom_stop_words = ["people", "like", "life", "one", "get",
   "never", "anyone", "day", "make", "else"]


## Models'  ROC-AUC and Test  Accuracy
________

Summary of Test and ROC Scores for Each Model for second iteration:

***CV stands for CountVectorizer
***TV stands for TfidVectorizer


| Model                | CV Test Score      | CV ROC Score       | TV Test Score       | TV ROC Score       |
| -----------          | -----------------  |--------------      |--------------       |----------          |
| Logistic Regression  | 0.785472972972973  | 0.785453216374269  | 0.7820945945945946  | 0.7811586257309941 |
| Bernoulli            | 0.7432432432432432 | 0.744517543859649  | 0.7432432432432432  | 0.744517543859649  |
| Multinomial          | 0.6942567567567568 | 0.6911549707602339 | 0.6976351351351351  | 0.6937134502923976 | 
| Random Forest        | 0.7820945945945946 | 0.7813413742690059 | 0.7736486486486487  | 0.7723866959064327 | 

Summary of Test and ROC Scores for each Model for first iteration:


| Model                | CV Test Score      | CV ROC Score       | TV Test Score       | TV ROC Score       |
| -----------          | -----------------  |--------------      |--------------       |----------          |
| Logistic Regression  | 0.7905405405405406 | 0.7902046783625731 | 0.7820945945945946  | 0.7808845029239766 |
| Bernoulli            | 0.7533783783783784 | 0.7546600877192983 | 0.7533783783783784  | 0.7546600877192983 |
| Multinomial          | 0.7010135135135135 | 0.6977339181286549 | 0.6959459459459459  | 0.6921600877192982 | 
| Random Forest        | 0.785472972972973  | 0.7840826023391813 | 0.7804054054054054  | 0.7792397660818713 | 


## Recommended Model: Benefit and Drawback
_________

Based on the highest ROC-AUC and test performances from the two iterations, we recommend the Logistic Regression model with the CountVectorizer transformer from the first iteration for social media startup's team to replicate and modify for their personality app, since the model was able to successfully distinguish between Showerthoughts and CasualConversation 79 percent of the time while the model was also able to correctly predict which new post will go to which subreddit 79 percent of the time as well. 

However, one drawback from this recommended model is that we observe case of overfitting since its training score was higher than test score. Overfitting in this context may mean that the model is overly confident in its predictions for this majority class, which is Showerthoughts, potentially making predictions that are overly specific to the training data and may not generalize well to new posts.


## Next Steps
__________

Here are following next steps to continue to improve the recommended model and investigate other findings from the four models:

* experiment with Logistic Regression's hyperparameters such as adjusting C to reduce model's overfitting
* experiment with other models' respective hyperparameters such as adjusting alpha and minimum sample leafs to reduce overfitting 
* investigate why overall, the test and ROC scores are generally similiar to one another 
* investigate why the Bernoulli model is the only one that was better at identifying CasualConversation than Showerthoughts
* investigate why Logistic Regression's test scores with TfidVectorizer from both iterations remain the same
* collect more unique subreddit posts to increase the diversity of words that our models can train on 