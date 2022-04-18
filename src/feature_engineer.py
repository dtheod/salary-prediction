import hydra
import numpy as np
import pandas as pd
from fuzzywuzzy import process
from omegaconf import DictConfig
from prefect import Flow, task
from prefect.engine.results import LocalResult
from prefect.engine.serializers import PandasSerializer

INTERMEDIATE_OUTPUT = LocalResult(
    "data/features_data/",
    location="{task_name}.csv",
    serializer=PandasSerializer("csv", serialize_kwargs={"index": False}),
)


@task
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, delimiter=",")


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
def job_feature(df: pd.DataFrame) -> pd.DataFrame:

    df = (
        df.assign(clean_job=lambda x: x["job"].str.lower())
        .assign(clean_job=lambda x: x["clean_job"].str.strip())
        .assign(
            clean_job=lambda x: x["clean_job"].str.replace(
                "sr", "senior", regex=False
            )
        )
        .assign(
            clean_job=lambda x: x["clean_job"].str.replace(
                "lead", "principal", regex=False
            )
        )
        .assign(
            job_counts=lambda x: x.groupby("clean_job")[
                "salary_usd"
            ].transform("count")
        )
        .assign(
            senior=lambda x: np.where(
                x["clean_job"].str.startswith("senior"), 1, 0
            )
        )
        .assign(
            lead=lambda x: np.where(
                x["clean_job"].str.startswith("lead"), 1, 0
            )
        )
        .assign(
            staff=lambda x: np.where(
                x["clean_job"].str.startswith("staff"), 1, 0
            )
        )
        .assign(
            clean_job=lambda x: x["clean_job"].str.replace(
                "senior", "", regex=False
            )
        )
        .assign(
            clean_job=lambda x: x["clean_job"].str.replace(
                "principal", "", regex=False
            )
        )
        .assign(
            clean_job=lambda x: x["clean_job"].str.replace(
                "staff", "", regex=False
            )
        )
        .assign(clean_job=lambda x: x["clean_job"].str.strip())
        .assign(
            intern=lambda x: x["clean_job"].apply(
                lambda x: 1 if "intern" == x.split(" ")[0] else 0
            )
        )
        .assign(
            assistant=lambda x: x["clean_job"].apply(
                lambda x: 1 if "assistant" == x.split(" ")[0] else 0
            )
        )
    )

    top_jobs = (
        df.sort_values("job_counts", ascending=False)
        .filter(["clean_job", "job_counts"])
        .drop_duplicates()
        .head(120)
    )
    top_jobs_list = top_jobs["clean_job"].unique()

    def fuzzy_match(row):
        if len(row) < 2:
            return row
        else:
            tup = process.extract(row, top_jobs_list, limit=1)[0]
            try:
                if tup[1] >= 90:
                    return tup[0]
                else:
                    return "Other"
            except IndexError:
                return "Other"

    df = df.assign(
        gender=lambda x: np.where(
            x["gender"].isin(
                [
                    "Other or prefer not to answer",
                    "nan",
                    "Prefer not to answer",
                ]
            ),
            "Other or prefer not to answer",
            x["gender"],
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
    def usa_func(row):
        if row in [
            "united states",
            "usa",
            "us",
            "united state of america",
            "united states of america",
            "unitedstates",
            "united sates of america",
            "america",
            "united state",
            "unites states",
            "united states of american",
            "united stated",
            "united sates",
            "the united states",
            "u s",
            "unite states",
            "united statea",
            "the us",
            "united stares",
        ]:
            return "united states"
        else:
            return row

    def uk_func(row):
        if row in [
            "united kingdom",
            "United Kingdomk",
            "uk",
            "scotland",
            "england, uk",
            "great britain",
            "england",
            "wales",
            "united kingdom (england)",
            "scotland, uk",
            "england, united kingdom",
            "uk (england)",
            "northern ireland",
        ]:
            return "united kingdom"
        else:
            return row

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
        .assign(clean_country=lambda x: x["clean_country"].apply(usa_func))
        .assign(clean_country=lambda x: x["clean_country"].apply(uk_func))
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


@task(result=INTERMEDIATE_OUTPUT)
def features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.filter(
        [
            "Id",
            "age",
            "Industry",
            "clean_country",
            "salary_usd",
            "clean_job",
            "senior",
            "lead",
            "staff",
            "intern",
            "assistant",
            "junior_experience",
            "mid_experience",
            "senior_experience",
            "old_experience",
        ]
    ).rename(
        columns={
            "Industry": "industry",
            "clean_country": "country",
            "salary_usd": "salary",
            "clean_job": "job",
            "Id": "id",
        }
    )

    return df


@hydra.main(config_path="../config", config_name="main")
def feature_data(config: DictConfig):

    with Flow("feature_data") as flow:

        df = (
            load_data(config.clean_data.path)
            .pipe(age_feature)
            .pipe(job_feature)
            .pipe(experience_feature)
            .pipe(country_feature)
        )
        df = features(df)

    flow.run()


if __name__ == "__main__":
    feature_data()
