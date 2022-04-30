import warnings
from typing import Any, Tuple
import bentoml
import hydra
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from omegaconf import DictConfig, OmegaConf
from prefect import Flow, task
from prefect.engine.results import LocalResult
from prefect.engine.serializers import PandasSerializer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from yellowbrick.contrib.wrapper import wrap
from yellowbrick.model_selection import LearningCurve
import wandb
warnings.filterwarnings("ignore")

FINAL_OUTPUT = LocalResult(
    "data/final/",
    location="{task_name}.csv",
    serializer=PandasSerializer("csv", serialize_kwargs={"index": False}),
)

CHECK_OUTPUT = LocalResult(
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
    wandb.log({"table": X_val})
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
    bentoml.sklearn.save("scaler", scaler)
    return scaler


@task
def apply_scaling(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    scaled = scaler.transform(df)
    return pd.DataFrame(scaled, columns=df.columns)


@task
def modeling(df_x: pd.DataFrame, df_y: pd.DataFrame) -> XGBRegressor:

    space = {
        "max_depth": hp.quniform("max_depth", 3, 18, 1),
        "gamma": hp.uniform("gamma", 1, 9),
        "reg_alpha": hp.quniform("reg_alpha", 40, 180, 1),
        "reg_lambda": hp.uniform("reg_lambda", 0, 1),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1),
        "min_child_weight": hp.quniform("min_child_weight", 0, 10, 1),
        "n_estimators": 1000,
        "seed": 0,
    }

    def objective(space):

        model = XGBRegressor(
            n_estimators=space["n_estimators"],
            max_depth=int(space["max_depth"]),
            gamma=space["gamma"],
            reg_alpha=int(space["reg_alpha"]),
            min_child_weight=int(space["min_child_weight"]),
            colsample_bytree=int(space["colsample_bytree"]),
        )
        X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(
            df_x, df_y, test_size=0.2
        )
        evaluation = [(X_train_scaled, y_train), (X_val_scaled, y_val)]

        model.fit(
            X_train_scaled,
            y_train,
            eval_set=evaluation,
            eval_metric="mae",
            verbose=False,
            early_stopping_rounds=50,
        )

        pred = model.predict(X_val_scaled)
        mae = mean_absolute_error(y_val, pred)
        print("SCORE:", mae)
        return {"loss": mae, "status": STATUS_OK}

    trials = Trials()
    best_hyper = fmin(
        fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials
    )

    model = XGBRegressor(
        n_estimators=1000,
        colsample_bytree=best_hyper["colsample_bytree"],
        gamma=best_hyper["gamma"],
        max_depth=int(best_hyper["max_depth"]),
        min_child_weight=best_hyper["min_child_weight"],
        reg_alpha=best_hyper["reg_alpha"],
        reg_lambda=best_hyper["reg_lambda"],
        eta=0.1,
        subsample=0.7,
    )

    model = wrap(model)
    model.fit(df_x, df_y)
    bentoml.xgboost.save(
        "booster_tree",
        model,
        booster_params={
            "disable_default_eval_metric": 1,
            "nthread": 2,
            "tree_method": "hist",
        },
    )
    return model


@task
def plot_training_curve(model: dict, df_x: pd.DataFrame, df_y: pd.DataFrame) -> None:

    sizes = np.linspace(0.3, 1.0, 10)
    cv = KFold(n_splits=5)
    visualizer = LearningCurve(model, cv=cv, scoring="r2", train_sizes=sizes)
    visualizer.fit(df_x, df_y)  # Fit the data to the visualizer
    visualizer.show(outpath="learning_curve.png")
    wandb.log({"training_curve": wandb.Image("learning_curve.png")})


@task(result=FINAL_OUTPUT)
def prediction(
    model: dict, df_x: pd.DataFrame, df_y: pd.DataFrame, df_x_val: pd.DataFrame
) -> pd.DataFrame:

    results = model.predict(df_x)
    comparison = pd.DataFrame(
        {"Predicted": results, "Actual": df_y, "id": df_x_val["id"]}
    )
    wandb.log({"r2_score": r2_score(comparison["Actual"], comparison["Predicted"])})
    wandb.log(
        {
            "mean_absolute_error": mean_absolute_error(
                comparison["Actual"], comparison["Predicted"]
            )
        }
    )
    return comparison


@task(result=CHECK_OUTPUT)
def reverse_engineer(preds_path: str, df: pd.DataFrame) -> pd.DataFrame:

    preds_and_actual = (
        pd.read_csv(preds_path)
        .merge(df, how="inner", on="id")
        .assign(diff=lambda df_: abs(df_["Actual"] - df_["Predicted"]))
    )

    return preds_and_actual


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
        X_train_scaled = apply_scaling(X_train, scaler)
        X_val_scaled = apply_scaling(X_val, scaler)

        # Model training
        reg2 = modeling(X_train_scaled, y_train)

        # Training Curve model
        # plot_training_curve(reg2, X_train_scaled, y_train)
        # mutual_information_viz(X_val, y_val)

        prediction(reg2, X_val_scaled, y_val, X_val)
        reverse_engineer(config.final_data.path, X_val)

        wandb_log(config)

    flow.run()


if __name__ == "__main__":
    predict()
