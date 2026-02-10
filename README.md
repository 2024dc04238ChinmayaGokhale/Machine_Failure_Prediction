# Machine Failure Prediction using Machine Learning

## a. Problem Statement

In modern manufacturing environments, unexpected machine failures lead
to production downtime, increased maintenance costs, and reduced
operational efficiency. The objective of this project is to predict
machine failure types using historical sensor and operational data by
applying multiple machine learning classification algorithms.

The problem is formulated as a multi-class classification task, where
the goal is to predict the type of failure (or no failure) based on
machine operating conditions.

------------------------------------------------------------------------

## b. Dataset Description

The dataset used in this project is related to predictive maintenance
and contains sensor readings and machine operational parameters.

### Input Features

-   Type: Machine type (L -- Low, M -- Medium, H -- High), encoded
    numerically
-   Air temperature \[K\]
-   Process temperature \[K\]
-   Rotational speed \[rpm\]
-   Torque \[Nm\]
-   Tool wear \[min\]

### Target Variable

-   Failure Type (multi-class):
    -   0 -- No Failure
    -   1 -- Tool Wear Failure
    -   2 -- Heat Dissipation Failure
    -   3 -- Power Failure
    -   4 -- Overstrain Failure
    -   5 -- Random Failure

Preprocessing steps included encoding categorical variables, feature
scaling using StandardScaler, and train-test splitting.

------------------------------------------------------------------------

## c. Models Used and Performance Evaluation


### Evaluation Metrics

-   Accuracy
-   AUC (Area Under ROC Curve)
-   Precision
-   Recall
-   F1 Score
-   Matthews Correlation Coefficient (MCC)

### Model Comparison Table

  ML Model Name              Accuracy   AUC    Precision   Recall   F1     MCC
  -------------------------- ---------- ------ ----------- -------- ------ ------
  Logistic Regression        0.97       0.93   0.97        0.98     0.97   0.75
  Decision Tree              0.95       0.91   0.94        0.95     0.94   0.70
  kNN                        0.94       0.89   0.93        0.94     0.93   0.68
  Naive Bayes                0.92       0.87   0.90        0.92     0.91   0.65
  Random Forest (Ensemble)   0.98       0.96   0.98        0.98     0.98   0.82
  XGBoost (Ensemble)         0.99       0.97   0.99        0.99     0.99   0.85

------------------------------------------------------------------------

## d. Model Performance Observations

  -----------------------------------------------------------------------
  ML Model Name                        Observation
  ------------------------------------ ----------------------------------
  Logistic Regression                  Performs well on majority classes
                                       but struggles with minority
                                       failure types due to linear
                                       boundaries.

  Decision Tree                        Captures non-linear relationships
                                       but prone to overfitting.

  kNN                                  Sensitive to scaling and class
                                       imbalance; performance degrades
                                       for large datasets.

  Naive Bayes                          Fast but limited by feature
                                       independence assumption.

  Random Forest                        Strong and stable performance due
                                       to ensemble learning.

  XGBoost                              Best overall model with superior
                                       handling of complex patterns and
                                       imbalance.
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## Conclusion

Ensemble models outperform individual classifiers. XGBoost achieved the
highest overall performance, making it the most suitable model for
predictive maintenance.

------------------------------------------------------------------------

## Deployment

The final model is deployed using Streamlit Community Cloud, enabling
real-time machine failure prediction through a web-based application.
