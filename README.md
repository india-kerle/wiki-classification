# Classifying toxic and obscene Wikipedia comments

### Objective
The project aims to identify, build and tune generalisable classifers to classify comments from Wikipedia talk pages into two categories: toxic and obscene. The data contained 20000 unique comments from discussions relating to user pages and articles. The comments in the train and test sets were hand labelled from folks at [_Jigsaw_](https://jigsaw.google.com/).

### Observations
* _inbalanced classes_ : there are 5018 labelled toxic comments while there are only 2622 labelled obscene comments. There are 2603 comments that are both toxic and obscene while there are 4963 comments that are neither.

* _Caps and !_: Comments that were toxic used capital letters more than comments that were neither or comments that were obscene. Both toxic and obscene comments used considerably more exclamation points than comments that were neither toxic nor obscene. I stacked these matrices with the document feature matrices but it resulted in much poorer performance across all five models.  

* _data wrangling_: There were no NAs or duplicates across the datasets - the only real data preprocessing that needed to occurs was preprocessing the text.

* _model tinkering_ : I attempted using SMOTE to address imbalanced classification but it in fact resulted in a lower weighted F1 score. I also fiddled with multilabel classification methods but it also resulted in a lower weighted F1 score. 

### Classification Methods Used
1. Logistic Regression
2. Random Forest
3. Decision Tree
4. Gaussian Naive Bayes
5. Support Vector Machine 

### Model Evaluation 
1. weighted F1 score due to target class imbalance

### Outputs

1. A test dataset containing a comment ID, comment and predicted toxic and obscene labels based on a tuned Support Vector Machine with a linear kernal (weighted F1 score: 91%) for obscene comments and a tuned Logistic Regression model for toxic comments (weighted F1 score: 89%). 
2. A diagnostics dataframe that includes weighted f1, weighted recall, accuracy, weighted, precision and roc auc scores from the five different models. 

