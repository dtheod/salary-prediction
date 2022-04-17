import numpy as np
import pandas as pd
from prefect import task
from prefect.engine.results import LocalResult
from prefect.engine.serializers import PandasSerializer

INTERMEDIATE_OUTPUT = LocalResult(
    "data/clean_data/",
    location="{task_name}.csv",
    serializer=PandasSerializer("csv", serialize_kwargs={"index": False}),
)


@task
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, delimiter="\t")


@task
def age_feature(df: pd.DataFrame) -> pd.DataFrame:

    df = df.assign(
        age=lambda x: np.where(
            x["age"].isin(["under 18", "18-24"]), "under 25", x["age"]
        )
    ).assign(
        age=lambda x: np.where(
            x["age"].isin(["55-64", "65 or over"]), "over_55", x["age"]
        )
    )
    return df


@task
def experience_feature(df: pd.DataFrame) -> pd.DataFrame:

    df = (
        df.assign(
            junior_experience=lambda x: np.where(
                x["years_field_experience"].isin(
                    ["1 year or less", "2 - 4 years"]
                ),
                1,
                0,
            )
        )
        .assign(
            mid_experience=lambda x: np.where(
                x["years_field_experience"].isin(["5-7 years"]), 1, 0
            )
        )
        .assign(
            senior_experience=lambda x: np.where(
                x["years_field_experience"].isin(
                    ["8 - 10 years", "11 - 20 years", "21 - 30 years"]
                ),
                1,
                0,
            )
        )
        .assign(
            old_experience=lambda x: np.where(
                x["years_field_experience"].isin(
                    ["31 - 40 years", "41 years or more"]
                ),
                1,
                0,
            )
        )
        .drop("years_field_experience", axis=1)
    )

    return df


@task
def country_feature(df: pd.DataFrame) -> pd.DataFrame:
    def netherlands_func(row):
        if row in ["netherlands", "the netherlands"]:
            return "netherlands"
        else:
            return row

    def newzealand_func(row):
        if row in ["new zealand", "nz"]:
            return "new zealand"
        else:
            return row

    df = (
        df.assign(
            clean_country=lambda x: x["country"].str.replace(
                ".", "", regex=False
            )
        )
        .assign(clean_country=lambda x: x["clean_country"].str.lower())
        .assign(clean_country=lambda x: x["clean_country"].str.strip())
        .assign(
            clean_country=lambda x: x["clean_country"].apply(netherlands_func)
        )
        .assign(
            clean_country=lambda x: x["clean_country"].apply(newzealand_func)
        )
        .assign(
            counts=lambda x: x.groupby("clean_country")[
                "salary_usd"
            ].transform("count")
        )
    )

    return df
