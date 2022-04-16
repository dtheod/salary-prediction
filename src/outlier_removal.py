import typing
import pandas as pd
import hydra
import numpy as np
from prefect import task, Flow, Parameter
from prefect.engine.results import LocalResult
from omegaconf import DictConfig
from prefect.engine.serializers import PandasSerializer
from rich.console import Console
console = Console()


@task
def load_data(path : str) -> pd.DataFrame:
    return pd.read_csv(path)

@task
def detect_remove(df: pd.DataFrame, ul:float, ll:float) -> pd.DataFrame:
    percentile1 = df['salary_usd'].quantile(ll)
    percentile99 = df['salary_usd'].quantile(ul)
    df = df.query("salary_usd > @percentile1 and salary_usd < @percentile99")
    return df


@hydra.main(config_path = "../config", config_name = "main")
def outlier_removal(config: DictConfig) -> None:

    with Flow("outlier_removal") as flow:
        
        df = load_data(config.clean_data.path)
        df = detect_remove(df, config.outliers.upper_limit,config.outliers.lower_limit)

    flow.run()

if __name__ == "__main__":
    outlier_removal()

















