# Social Media Ads Classification
## Scope of Study
The classification of social media ads involves analyzing the ads to classify whether the target audience will buy the product or not. It's a valuable use case for data science in marketing. By classifying social media ads, businesses can find the most profitable customers for their products, those who are more likely to make a purchase.

Sometimes, a product may not be suitable for all people based on age and income. For example, a person between the ages of 20 and 25 may be more inclined to spend on smartphone covers than a person between the ages of 40 and 45. Similarly, a high-income person can afford to spend more on luxury goods than a low-income person. By analyzing social media ads, businesses can determine whether a person will buy their product or not.

# Dataset Description
The dataset used for this task contains data about a product’s social media advertising campaign. It includes features like:
Age of the target audience
Estimated salary of the target audience
Whether the target audience has purchased the product or not

The dataset is downloaded from Kaggle.

## Data Exploration and Visualization
Data Exploration and Visualization is done using the 
pandas 
numpy
matplotli
seaborn

## Model Building and Evaluation
Model Building and Evaluation done using 
DecisionTree Classifier (The accuracy score = 0.82)
RandomForestClassifier  (The accuracy score = 0.90)
Naive Bayes             (The accuracy score = 0.93)
Support Vector Machine  (The accuracy score = 0.75)
Logistic Regression     (The accuracy score = 0.68)

The best fit model is Naive Bayes which has better accuiracy of 0.93

## Insights
The target audience's over-45 demographic is more interested in buying the goods
The target group who earn more than 90,000rs per month are more likely to buy the goods.
