# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

### Developer
This project was developed by J. Deonarine to meet the requirements of the course "Machine Learning DevOps", part of the Udacity Data Science Nanodegree and a requirement for Western Governors Universtiy bachelor's in Data Analytics.

### Date
07/28/2025

### Version
1.0.0

### Type
The model used was trained using the pre-trained AdaBoost classifier from the scikit-learn framework.

## Intended Use
The project is intended to develop/showcase the author's ability to write in and use the various dependant libraries and frameworks.

The pipeline ingests and cleans data to fit to a ML model.  The pipeline itself is intended to be deployed as Heroku web app and combined with an API in FastAPI.  When live, the API is in a CI/CD framework.

The model itself is intended to determine, based on other census data points, whether the individual makes more or less than $50,000 per year.

## Training Data
The training data set was retrieved from the US Census Income Data Set obtained from the UCI Machine Learning Repository.  The training set is an 80% split.

## Evaluation Data
The remaining 20% of the data was used as a test set.

## Metrics
* Precision - 0.7324
* Recall - 0.5990
* F1 score - 0.6590

Hyperparameter optimization was conducted using GridSearchCV and found a learning rate of 1 and 500 n_estimators to be the ideal settings.  

## Ethical Considerations
Census data, especially concerning income, has unavoidable bias.  Care should be used when utilizing this predictive model in way that could be construed as descriminatory against protected classes.

## Caveats and Recommendations
Any attempts to summarize or predict complex human behavior based on predictions bares inherent risk.  Also, this dataset is dated (2019) and will not reflect the major economic changes that have occurred since then.