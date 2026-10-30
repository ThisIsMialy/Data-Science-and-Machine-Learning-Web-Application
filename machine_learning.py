"""
machine_learning.py
ITS72604 Assessment 2

TASK 2:
7-Step Machine Learning Framework using Scikit-Learn to predict `Grid_Status_Label`

GOAL:
Find the best model for the dataset 'cleaned_ev_dataset.csv'
"""

# LIBRARIES
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

# SCIKIT LEARN
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def save_model_joblib(model) -> None:
    """
    Procedure to export the highest-performing trained model
    :param model: Estimator (Model)
    :return: None
    """
    filepath = 'assets/ev_grid_label_model.joblib'
    print(f"Your model has been saved at {filepath}")
    joblib.dump(model, filepath)

    return None


def get_metrics(y_test, y_pred) -> dict:
    """
    Called after .predict() of each tuned model
    Calculates metrics based on predictions of the models, rounded to 4 decimal points
    Get accuracy, precision, recall, f1 score, confusion matrix and classification_report
    Prints Classification Report

    :param y_test: Actual labels
    :param y_pred: Predicted labels after .predict()
    :return: Dictionary in the form : {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_m': confusion_m,
        'classification_report': class_report
        }
    """
    accuracy = round(accuracy_score(y_test, y_pred), 4) * 100
    precision = round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4) * 100
    recall = round(recall_score(y_test, y_pred, average='weighted'), 4) * 100
    f1 = round(f1_score(y_test, y_pred, average='weighted'), 4) * 100
    confusion_m = confusion_matrix(y_test, y_pred)

    # Print in percentage the accuracy, precision, recall and f1 score
    print(f'Accuracy: {accuracy}% / Precision: {precision}% / Recall: {recall}% / F1: {f1}%')
    # Print confusion matrix
    print("Confusion Matrix")
    print(confusion_m)
    print('\n')
    # get and print classification report
    class_report = classification_report(y_test, y_pred, zero_division=0)
    print(class_report)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_m': confusion_m,
        'classification_report': class_report
    }


def print_results(score) -> None:
    """
    Print best results of GridSearchCV
    GridSearchCV is used to test and score the hyperparemeters of the models
    :param score: score from GridSearchCV
    :return: None
    """
    means = score.cv_results_['mean_test_score']
    sd = score.cv_results_['std_test_score']

    print(f'Grid Search Best Score: {round(score.best_score_, 5)} \n BEST PARAMS: {score.best_params_} \n')

    ### PRINTS all scores
    # for mean, std, params in zip(means, sd, score.cv_results_['params']):
    #     print(f'{round(mean, 2)} (+/-{round(std * 2, 2)}) for {params}')

def plot_heatmap(confusion_m, class_labels, ax):
    """
    Create and return a heatmap given the confusion matrix of a model's predictions
    :param confusion_m: confusion matrix from metrics
    :param class_labels: Grid Status Labels
    :param ax: axes position to place the heatmap
    :return: Seaborn Confusion Matrix Heatmap with index
    """
    return sns.heatmap(
        confusion_m,
        annot=True,
        fmt='d',
        xticklabels=class_labels,
        yticklabels=class_labels,
        cmap='Blues',
        cbar=True,
        ax=ax
    )


def compare_best_models(models,
                        model_names,
                        X_test,
                        y_test) -> Pipeline | None:
    """
    Compare the best models based on accuracy
    All best models are stored in array `models`
    Predict using X_test dataset and compare the predictions to y_test
    Call the function `get_metrics` to calculate the score of each model
    Create and plot heatmaps for each model using the `for` loop
    Store the best_accuracy and best best_model_overall

    :param models: all trained models
    :param model_names: names of models in order of their appearance
    :param X_test: datasets to be tested on
    :param y_test: Actual values of labels (X_test)
    :return: Pipeline containing model
    """
    best_accuracy = 0
    best_model_overall = None

    # Plot heatmaps in the same grid for comparison
    fig, axes = plt.subplots(1, len(models), figsize=(32, 8))

    # Plot heatmaps for the confusion matrix of each model
    for ax, (name, model) in enumerate(zip(model_names, models)):
        y_pred = model.predict(X_test)
        class_labels = model.classes_
        print(f"======================================================================")
        print(f"------- {name} PERFORMANCE ------")
        metrics = get_metrics(y_test, y_pred)

        # store the best model based on accuracy of predictions
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            best_model_overall = model

        # call function plot_heatmap to get the seaborn heat map
        plot_heatmap(metrics['confusion_m'], class_labels, axes[ax])
        axes[ax].set_title(name)
        axes[ax].set_xlabel("Predicted Grid Status")
        axes[ax].set_ylabel("True Grid Status")

    plt.tight_layout()
    plt.show()
    print(f"The best model is: {best_model_overall}")

    return best_model_overall


def apply_decision_tree_classifier(X_train, y_train) -> Pipeline:
    """
    Pipeline has:
    1. StandardScaler(), to scale the training data that passes through
    2. Estimator = DecisionTreeClassifier() at random state = 42

    Stratified K-fold is applied to the training dataset, split into 5
    Then calculate baseline of the model using `cross_val_score` to get Mean accuracy

    Tune the hyperparameters for the estimator
    Then apply cross-validation and call GridSearchCV to find the best hyperparameters

    Call `print_results` to determine the best scoring hyperparameters

    :param X_train: Training data
    :param y_train: Training labels
    :return: Best estimator returned as Pipeline with DecisionTreeClassifier()
    """
    dtc_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', DecisionTreeClassifier(
            random_state=42,
        ))
    ])

    # Perform cross validation on the training datasets
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(estimator=dtc_pipe, X=X_train, y=y_train, cv=cv, scoring='accuracy')

    print("----- DECISION TREE CLASSIFIER -----")
    print("BASELINE: Cross Validation Scores")
    print(f"Fold Accuracy Scores: {scores}")
    print(f"Mean Accuracy: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

    param_grid = {
        'classifier__criterion': ['gini', 'log_loss'],
        'classifier__max_features': ['sqrt', 'log2'],
        'classifier__splitter': ['best', 'random']
    }

    grid_search_dtc = GridSearchCV(estimator=dtc_pipe,
                                   param_grid=param_grid,
                                   cv=cv)

    grid_search_dtc.fit(X_train, y_train)

    print_results(grid_search_dtc)
    best_dtc_model = grid_search_dtc.best_estimator_
    return best_dtc_model


def apply_support_vector_classifier(X_train, y_train) -> Pipeline:
    """
    Pipeline has:
    1. StandardScaler(), to scale the training data that passes through
    2. Estimator = SVC() at random state = 42 [Support Vector Classifier]

    Stratified K-fold is applied to the training dataset, split into 5
    Then calculate baseline of the model using `cross_val_score` to get Mean accuracy

    Tune the hyperparameters for the estimator
    Then apply cross-validation and call GridSearchCV to find the best hyperparameters

    Call `print_results` to determine the best scoring hyperparameters

    :param X_train: Training data
    :param y_train: Training labels
    :return: Best estimator returned as Pipeline with SVC()
    """
    svc_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(
            random_state=42,
        ))
    ])

    # Perform cross validation on the training datasets
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(estimator=svc_pipe, X=X_train, y=y_train, cv=cv, scoring='accuracy')

    print("----- SUPPORT VECTOR CLASSIFIER -----")
    print("BASELINE: Cross Validation Scores")
    print(f"Fold Accuracy Scores: {scores}")
    print(f"Mean Accuracy: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

    param_grid = {
        'classifier__kernel': ['linear', 'rbf', 'poly'],
        'classifier__C': [0.5, 1, 1.5, 2, 10, 50, 100],
        'classifier__gamma': ['scale', 'auto']
    }

    grid_search_svc = GridSearchCV(estimator=svc_pipe,
                                   param_grid=param_grid,
                                   cv=cv)

    grid_search_svc.fit(X_train, y_train)

    print_results(grid_search_svc)
    best_svc_model = grid_search_svc.best_estimator_
    return best_svc_model




def apply_logistic_regression(X_train, y_train) -> Pipeline:
    """
    Pipeline has:
    1. StandardScaler(), to scale the training data that passes through
    2. Estimator = LogisticRegression() at random state = 42

    Stratified K-fold is applied to the training dataset, split into 5
    Then calculate baseline of the model using `cross_val_score` to get Mean accuracy

    Tune the hyperparameters for the estimator
    Then apply cross-validation and call GridSearchCV to find the best hyperparameters

    Call `print_results` to determine the best scoring hyperparameters

    :param X_train: Training data
    :param y_train: Training labels
    :return: Best estimator returned as Pipeline with LogisticRegression()
    """

    # Apply Standard Scaler when model is called
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            random_state=42)
        )
    ])

    # Define Cross-Validation method
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(estimator=lr_pipe, X=X_train, y=y_train, cv=cv, scoring='accuracy')

    print("----- LOGISTIC REGRESSION -----")
    print("BASELINE: Cross Validation Scores")
    print(f"Fold Accuracy Scores: {scores}")
    print(f"Mean Accuracy: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

    # Set different Hyperparameters for tuning
    param_grid = {
        'classifier__C': [0.7],
        'classifier__max_iter': [1000, 5000]
    }

    # Apply a GridSearchCV to go through all hyperparameters
    grid_search = GridSearchCV(
        estimator=lr_pipe,
        param_grid=param_grid,
        cv=cv
    )
    # Train the model on the training
    grid_search.fit(X_train, y_train)

    print_results(grid_search)
    best_lr_model = grid_search.best_estimator_
    return best_lr_model


def apply_random_forest_classifier(X_train, y_train) -> Pipeline:
    """
    Estimator = RandomForestClassifier() at random state = 42
    [No scaling required for RandomForestClassifier()]

    Stratified K-fold is applied to the training dataset, split into 5
    Then calculate baseline of the model using `cross_val_score` to get Mean accuracy

    Tune the hyperparameters for the estimator
    Then apply cross-validation and call GridSearchCV to find the best hyperparameters

    Call `print_results` to determine the best scoring hyperparameters

    :param X_train: Training data
    :param y_train: Training labels
    :return: Best estimator returned as RandomForestClassifier()
    :return: RandomForestClassifier()
    """
    rfc_pipe = RandomForestClassifier(
        max_features='sqrt',
        random_state=42,
        criterion='gini',
        max_leaf_nodes=None,
        max_samples=None
    )

    # Define Cross-Validation method
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(estimator=rfc_pipe, X=X_train, y=y_train, cv=cv, scoring='accuracy')


    print("----- RANDOM FOREST CLASSIFICATION -----")
    print("BASELINE: Cross Validation Scores")
    print(f"Fold Accuracy Scores: {scores}")
    print(f"Mean Accuracy: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

    # OTHER POSSIBLE PARAMETERS:
    #'max_depth': [2, 4, 6, 8]
    # 'min_samples_split': [2, 4],
    # 'min_samples_leaf': [2, 4],
    param_grid = {
        "max_features": [5, 7, 9]
    }

    grid_search_rfc = GridSearchCV(
        estimator=rfc_pipe,
        param_grid=param_grid,
        cv=cv
    )

    # Train the model on the training datasets and apply grid search
    grid_search_rfc.fit(X_train, y_train)
    print_results(grid_search_rfc)
    ## get the best performer
    best_rfc_model = grid_search_rfc.best_estimator_
    return best_rfc_model


def main() -> None:
    """
    Open CSV file with Pandas and convert to DataFrame.
    One hot encoding of `Grid_Status_Label` with `OneHotEncoder`

    To train data (X):
    Drop columns that may be irrelevant and/or calculated from other features
    ['Session_ID', 'User_Type', 'Efficiency_Group',
    'Carbon_Savings_kgCO2', 'Power_kW', 'Grid_Status_Label']

    To test data (y): Store only Grid_Status_Label

    Runs ML algorithms on the split datasets
    Evaluate performance of each model and compare their accuracies
    Export the best model (one model) with joblib
    :return: None
    """
    df = pd.read_csv('assets/cleaned_ev_dataset.csv')

    charger_type_ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    encoded_data = charger_type_ohe.fit_transform(df[['Charger_Type']])

    feature_name = charger_type_ohe.get_feature_names_out(['Charger_Type'])
    print("Created new Features: ", feature_name)

    # Create a new Pandas dataframe to store encoded data and new feature names
    encoded_df = pd.DataFrame(
        encoded_data,
        columns=feature_name,
    )

    # Concatenate the new df with old df and drop Charger_Type from the dataframe
    final_df = pd.concat([df.drop(columns=['Charger_Type']), encoded_df], axis=1)

    # Drop features that do not matter: Session ID, Grid Status
    # Drop features that were calculated from other features : Power, Carbon_Savings
    X = final_df.drop(columns=['Session_ID', 'User_Type', 'Efficiency_Group',
                               'Carbon_Savings_kgCO2', 'Power_kW', 'Grid_Status_Label'])
    y = final_df['Grid_Status_Label']


    print("Training data........ \n")

    # Split data into 80% Training, 20% Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Call all classifiers (models) and get all only the best model after hyperparameter tuning
    best_lr = apply_logistic_regression(X_train, y_train)

    best_rfc = apply_random_forest_classifier(X_train, y_train)

    best_svc = apply_support_vector_classifier(X_train, y_train)

    best_dtc = apply_decision_tree_classifier(X_train, y_train)

    # Create array to store all the best estimators
    models = [best_lr, best_rfc, best_svc, best_dtc]
    model_names = [
        "Logistic Regression",
        "Random Forest Classifier",
        "Support Vector Classifier",
        "Decision Tree Classifier",
    ]
    # Only save the best model after comparison
    # Model persistence: use `joblib` to store highest-performing model
    best_model_overall = compare_best_models(models, model_names, X_test, y_test)
    save_model_joblib(best_model_overall)

    return None


# START HERE
if __name__ == '__main__':
    main()
