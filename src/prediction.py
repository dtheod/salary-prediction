from typing import Any, Tuple

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from prefect import Flow, task
from prefect.engine.results import LocalResult
from prefect.engine.serializers import PandasSerializer
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                              VotingRegressor)
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from yellowbrick.model_selection import LearningCurve
#from yellowbrick.target import FeatureCorrelation

import wandb

FINAL_OUTPUT = LocalResult(
    "data/final/",
    location="{task_name}.csv",
    serializer=PandasSerializer("csv", serialize_kwargs={"index": False}),
)


@task
def initialize_wandb(config: DictConfig):
    wandb.init(
        project="salary_prediction",
        config=OmegaConf.to_object(config),
        reinit=True,
        mode=config.wandb_mode,
    )


@task
def load_features(path: str) -> pd.DataFrame:
    data = pd.read_csv(path, delimiter=",")
    return data


@task
def train_val_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    X = df.drop("salary", axis=1)
    y = df["salary"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    wandb.log({"table": X})
    return X_train, X_val, y_train, y_val


@task
def one_hot_encoding(df_train: pd.DataFrame) -> pd.DataFrame:

    all_data_types = list(df_train.dtypes)
    columns = list(df_train)
    for data_type, colum in zip(all_data_types, columns):
        if str(data_type) == "object":
            df_train[colum] = df_train[colum].astype("category")
    return pd.get_dummies(df_train)


@task
def standard_scaling(df: pd.DataFrame) -> Any:
    scaler = StandardScaler()
    scaler.fit(df)
    return scaler


@task
def apply_scaling(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    scaled = scaler.transform(df)
    return pd.DataFrame(scaled, columns=df.columns)


@task
def modeling(df_x: pd.DataFrame, df_y: pd.DataFrame) -> RandomForestRegressor:

    alpha = 0.7
    l1_ratio = 0.7
    learning_rate = 0.01
    n_estimators = 500

    reg1 = GradientBoostingRegressor(
        random_state=1, learning_rate=learning_rate, n_estimators=n_estimators
    )
    reg3 = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=1)

    ereg = VotingRegressor(estimators=[("gb", reg1), ("en", reg3)])
    ereg = ereg.fit(df_x, df_y)
    return ereg


@task
def plot_training_curve(model: dict, df_x: pd.DataFrame, df_y: pd.DataFrame) -> None:

    sizes = np.linspace(0.3, 1.0, 10)
    cv = KFold(n_splits=5)
    visualizer = LearningCurve(model, cv=cv, scoring="r2", train_sizes=sizes, n_jobs=3)
    visualizer.fit(df_x, df_y)  # Fit the data to the visualizer
    visualizer.show(outpath="learning_curve.png")
    wandb.log({"training_curve": wandb.Image("learning_curve.png")})


# @task
# def mutual_information_viz(features: np.ndarray, target: np.ndarray) -> None:
#     discrete = [False for _ in range(len(features))]
#     discrete[1] = True

#     visualizer = FeatureCorrelation(method="mutual_info-regression", labels=features)
#     visualizer.fit(features, target, discrete_features=discrete, random_state=0)
#     visualizer.show(outpath="mututal_information.png")
#     wandb.log({"mutual_information": wandb.Image("mututal_information.png")})


@task(result=FINAL_OUTPUT)
def prediction(model: dict, df_x: pd.DataFrame, df_y: pd.DataFrame) -> pd.DataFrame:

    results = model.predict(df_x)
    comparison = pd.DataFrame({"Predicted": results, "Actual": df_y})
    wandb.log({"r2_score": r2_score(comparison["Actual"], comparison["Predicted"])})
    return comparison


@task
def wandb_log(config: DictConfig):

    # log data
    wandb.log_artifact(config.raw_data.path, name="raw_data", type="data")
    wandb.log_artifact(config.clean_data.path, name="clean_data", type="data")
    wandb.log_artifact(config.feature_data.path, name="features", type="data")


@hydra.main(config_path="../config", config_name="main")
def predict(config: DictConfig) -> None:

    with Flow("prediction") as flow:
        initialize_wandb(config)

        # Load the features dataset
        df = load_features(config.feature_data.path)

        # Perform One hot encoding on all categorical variables
        df = one_hot_encoding(df)

        # Split the data into train and validation sets
        X_train, X_val, y_train, y_val = train_val_split(df)

        # Instantiate Standard scaling using only the train data
        scaler = standard_scaling(X_train)

        # Apply scaling for train and val data
        X_train = apply_scaling(X_train, scaler)
        X_val = apply_scaling(X_val, scaler)

        # Model training
        reg2 = modeling(X_train, y_train)

        # Training Curve model
        plot_training_curve(reg2, X_train, y_train)
        # mutual_information_viz(X_val, y_val)

        prediction(reg2, X_val, y_val)

        wandb_log(config)

    flow.run()


if __name__ == "__main__":
    predict()
