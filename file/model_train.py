import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from imblearn.over_sampling import RandomOverSampler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

def load_data(data_path, target):
    df = pd.read_csv(data_path)
    y = df[target]
    X = df.drop(columns=[target])
    return X, y

def split_data(X, y, test_size=0.2, random_state=40):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def oversample_data(X_train, y_train, random_state=40):
    over_sampler = RandomOverSampler(random_state=random_state)
    X_train_over, y_train_over = over_sampler.fit_resample(X_train, y_train)
    return X_train_over, y_train_over

def run_experiment(X_train, y_train, X_test, y_test, params):
    with mlflow.start_run():
        clf = make_pipeline(
            SimpleImputer(),
            RandomForestClassifier(random_state=40)
        )

        model = GridSearchCV(
            clf,
            param_grid=params,
            cv=5,
            n_jobs=-1,
            verbose=1
        )

        model.fit(X_train, y_train)

        acc_train = model.score(X_train, y_train)
        acc_test = model.score(X_test, y_test)

        print(f"Training Accuracy: {acc_train}")
        print(f"Test Accuracy: {acc_test}")

        mlflow.log_param("params", params)
        mlflow.log_metric("acc_train", acc_train)
        mlflow.log_metric("acc_test", acc_test)

        mlflow.sklearn.log_model(model, "model")

def main(args):
    X, y = load_data(args.data_path, args.target)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_over, y_train_over = oversample_data(X_train, y_train)

    params = {
        "simpleimputer__strategy": ["mean", "median"],
        "randomforestclassifier__n_estimators": range(25, 100, 25),
        "randomforestclassifier__max_depth": range(10, 50, 10)
    }

    run_experiment(X_train_over, y_train_over, X_test, y_test, params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a model")
    parser.add_argument("--data_path", type=str, help="Path to the dataset")
    parser.add_argument("--target", type=str, help="Target variable name")
    
    args = parser.parse_args()
    
    main(args)