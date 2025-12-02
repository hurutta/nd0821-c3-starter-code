# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
RandomForestClassifier trained to predict a binary label from tabular features.  
Framework: scikit-learn.

## Intended Use
For educational and exploratory ML classification tasks.

## Training Data
Adult Census Income dataset containing demographic and work-related attributes:  
`age, workclass, fnlgt, education, education-num, marital-status, occupation, relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, salary`.

## Evaluation Data
Held-out split from the same dataset, same schema and preprocessing applied.

## Metrics
Metrics: Precision, Recall, F1 
Used to evaluate binary classification quality for predicting income.
Precision: 0.7555555555555555, Recall: 0.643860720830788, F1: 0.6952506596306068

## Ethical Considerations
Dataset contains sensitive attributes (race, sex, marital status).  
Model may amplify historical bias; not appropriate for decision-making affecting rights or opportunities.

## Caveats and Recommendations
Model performance depends on feature processing and dataset representativeness.  
Not robust to deployment in real-world socioeconomic settings, use only for learning and experimentation.