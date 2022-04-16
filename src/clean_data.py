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

INTERMEDIATE_OUTPUT = LocalResult(
    "data/clean_data/",
    location="{task_name}.csv",
    serializer=PandasSerializer("csv", serialize_kwargs={"index": False}),
)

@task
def load_data(path : str) -> pd.DataFrame:
    return pd.read_csv(path, delimiter = "\t")

@task
def rename_selection(df:pd.DataFrame) -> pd.DataFrame:
    df = (
        df
        .rename(columns = {'Timestamp':'timestamp','How old are you?':'age', 'Job title':'job',
                            'Additional context on job title': 'job_context', 'Annual salary':'salary',
                            'Other monetary comp': 'other_monetary', 'Currency':'currency',
                            'Currency - other':'currency_other', 'Additional context on income': 'income_context',
                            'Country':'country', 'State':'state', 'City':'city',
                            'Overall years of professional experience': 'years_epxerience',
                            'Years of experience in field': 'years_field_experience',
                            'Highest level of education completed': 'education', 'Gender': 'gender', 'Race':'race'})
                .drop(['job_context', 'other_monetary', 'income_context', 'state', 'timestamp'], axis = 1)
                .assign(salary = lambda x: pd.to_numeric(x['salary'].str.replace(",", "")))
                .reset_index()
                .rename(columns = {'index':'Id'})
                .fillna('nan')
    )
    return df

@task(result = INTERMEDIATE_OUTPUT)
def currency_conversion(df:pd.DataFrame) -> pd.DataFrame:
    
    ecb_fx_url = 'https://proxy.hxlstandard.org/data/490a41/download/ECB_FX_USD-quote.csv'

    fxusd_rates = (
        pd.read_csv(ecb_fx_url)
        .iloc[1:, :]
        .head(1)
        .transpose()
        .rename(columns = {1: 'rate'})
        .reset_index()
        .query("index != 'Date'")
        .rename(columns = {'index':'pair'})
        .assign(rate = lambda x: pd.to_numeric(x['rate']).round(5))
        .assign(conv_pair = 'USD')
    )

    df = (
            df
            .assign(currency = lambda x: np.where(x['currency'] == 'AUD/NZD','AUD', x['currency']))
            .merge(fxusd_rates, how = 'left', left_on = 'currency', right_on = 'pair')
            .assign(pair = lambda x: x['pair'].fillna('nan'))
            .assign(salary_usd = lambda x: np.where(x['pair'] != 'nan', x['salary'] * x['rate'], x['salary']))
            .assign(new_currency = lambda x: np.where(x['pair'] != 'nan', x['conv_pair'], x['currency']))
            .assign(currency = lambda x: np.where(x['currency'] == 'AUD/NZD','AUD', x['currency']))
            .drop(['pair', 'rate', 'conv_pair'], axis = 1)
            .query("currency != 'Other'")
    )

    return df

@hydra.main(config_path="../config", config_name="main")
def clean_data(config: DictConfig):

    with Flow("clean_data") as flow:

        df = (
            load_data(config.raw_data.path)
            .pipe(rename_selection)
            .pipe(currency_conversion)            
        )
    flow.run()


if __name__ == "__main__":
    clean_data()