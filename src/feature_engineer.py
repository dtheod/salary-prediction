import pandas as pd
from prefect import task
from prefect.engine.results import LocalResult
from prefect.engine.serializers import PandasSerializer

print("Change")

INTERMEDIATE_OUTPUT = LocalResult(
    "data/clean_data/",
    location="{task_name}.csv",
    serializer=PandasSerializer("csv", serialize_kwargs={"index": False}),
)


@task
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, delimiter="\t")


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
