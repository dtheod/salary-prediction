import warnings

import bentoml
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from prefect import Flow, task
from prefect.engine.results import LocalResult
from prefect.engine.serializers import PandasSerializer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from yellowbrick.cluster import KElbowVisualizer

warnings.filterwarnings("ignore")

INTERMEDIATE_OUTPUT = LocalResult(
    "data/features_data/",
    location="{task_name}.csv",
    serializer=PandasSerializer("csv", serialize_kwargs={"index": False}),
)


class CustomFeature(BaseEstimator, TransformerMixin):
    def __init__(self):
        print("Initialising education feature")

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print("feature transform")
        X = education_feature.run(X)
        X = experience_feature.run(X)
        X = industry_feature.run(X)
        X = job_feature.run(X)
        return X


@task
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, delimiter=",")


@task
def age_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(
        age=lambda x: np.where(x["age"].isin(["under 18", "18-24"]), "under 25", x["age"])
    ).assign(
        age=lambda x: np.where(
            x["age"].isin(["55-64", "65 or over"]), "over_55", x["age"]
        )
    )
    return df


@task
def education_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(
        education=lambda df_: df_["education"].replace(
            {
                "High School": 10,
                "Some college": 12,
                "College degree": 14,
                "Master's degree": 16,
                "PhD": 18,
                "Professional degree (MD, JD, etc.)": 22,
            }
        )
    ).assign(education=lambda df_: df_["education"].mode()[0])
    return df


@task
def job_feature(df: pd.DataFrame) -> pd.DataFrame:

    df = (
        df.pipe(
            lambda df_: df_.assign(
                clean_job=(
                    df_["job"]
                    .str.lower()
                    .str.strip()
                    .str.replace("sr", "senior", regex=False)
                    .str.replace("lead", "principal", regex=False)
                )
            )
        )
        .pipe(
            lambda df_: df_.assign(
                senior=np.where(df_["clean_job"].str.startswith("senior"), 1, 0),
                principal=np.where(df_["clean_job"].str.startswith("principal"), 1, 0),
                staff=np.where(df_["clean_job"].str.startswith("staff"), 1, 0),
                assistant=np.where(df_["clean_job"].str.startswith("assistant"), 1, 0),
                intern=np.where(df_["clean_job"].str.startswith("intern"), 1, 0),
            )
        )
        .pipe(
            lambda df_: df_.assign(
                clean_job=(
                    df_["clean_job"]
                    .str.replace("senior", "", regex=False)
                    .str.replace("principal", "", regex=False)
                    .str.replace("junior", "", regex=False)
                    .str.replace("staff", "", regex=False)
                    .str.strip()
                )
            )
        )
        .drop("job", axis=1)
        .rename(columns={"clean_job": "job"})
    )

    return df


@task
def gender_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(
        gender=lambda df_: np.where(
            df_["gender"].isin(
                [
                    "Other or prefer not to answer",
                    "nan",
                    "Prefer not to answer",
                ]
            ),
            "Other or prefer not to answer",
            df_["gender"],
        )
    )
    return df


@task
def experience_feature(df: pd.DataFrame) -> pd.DataFrame:

    df = df.assign(
        years_field_experience=lambda df_: df_["years_field_experience"].replace(
            {
                "1 year or less": 1,
                "2 - 4 years": 3,
                "5-7 years": 6,
                "8 - 10 years": 9,
                "11 - 20 years": 15,
                "21 - 30 years": 25,
                "31 - 40 years": 35,
                "41 years or more": 40,
            }
        )
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

    def europe_func(row):
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
            "germany",
            "ireland",
            "netherlands",
            "the netherlands",
            "france",
            "sweden",
            "belgium",
            "spain",
            "austria",
            "finland",
            "italy",
            "denmark",
        ]:
            return "europe"
        else:
            return row

    def oceania_func(row):
        if row in ["new zealand", "nz", "australia"]:
            return "oceania"
        else:
            return row

    df = (
        df.pipe(
            lambda df_: df_.assign(
                clean_country=(
                    df_["country"]
                    .str.replace(".", "", regex=False)
                    .str.lower()
                    .str.strip()
                    .apply(usa_func)
                    .apply(europe_func)
                    .apply(oceania_func)
                )
            )
        )
        .assign(
            counts=lambda df_: df_.groupby("clean_country")["salary_usd"].transform(
                "count"
            )
        )
        .assign(
            clean_country=lambda df_: np.where(
                df_["counts"] >= 10, df_["clean_country"], "Other"
            )
        )
    )

    return df


@task
def industry_feature(df: pd.DataFrame) -> pd.DataFrame:
    def topn(ser, n=22, default="other"):
        counts = ser.value_counts()
        return ser.where(ser.isin(counts.index[:n]), default)

    df = df.pipe(lambda df_: df_.assign(Industry=df_["Industry"].str.lower().pipe(topn)))
    return df.rename(columns={"Industry": "industry"})


# @task
# def salary_features(df: pd.DataFrame) -> pd.DataFrame:
#     df = (
#         df.assign(counts=lambda x: x.groupby("job")["salary"].transform("count"))
#         .assign(job=lambda x: np.where(x["counts"] > 2, x["job"], "Other"))
#         .query("job != 'Other'")
#         .assign(
#             job_salary_mean=lambda x: x.groupby(["job"])["salary"].transform("mean")
#         )
#         .assign(
#             job_salary_std=lambda x: x.groupby(["job"])["salary"].transform("std")
#         )
#     )
#     return df


@task
def clustering_feature(df: pd.DataFrame) -> pd.DataFrame:
    vectorizer = TfidfVectorizer(stop_words={"english"})
    tfidf = vectorizer.fit(df.job.values.tolist())
    bentoml.sklearn.save("tfidf", tfidf)

    X = tfidf.transform(df.job.values.tolist())
    true_k = 8
    model1 = KMeans(init="k-means++", max_iter=200, n_init=10)
    visualizer = KElbowVisualizer(model1, k=(4, 12))
    visualizer.fit(X)
    visualizer.show("elbow_method.png")
    model = KMeans(n_clusters=true_k, init="k-means++", max_iter=200, n_init=10)
    model.fit(X)
    bentoml.sklearn.save("kmeans", model)

    labels = model.labels_
    labels = ["c" + str(label) for label in labels]
    df["cluster"] = labels
    return df


@task
def feature_estimators(df: pd.DataFrame) -> pd.DataFrame:
    my_pipe = Pipeline(steps=[("custom_pipe", CustomFeature())])
    my_pipe.fit(df)
    print("--PASSED--")
    bentoml.sklearn.save("custom_transform_pipe", my_pipe)
    return df


@task(result=INTERMEDIATE_OUTPUT)
def features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.filter(
        [
            "Id",
            "age",
            "clean_country",
            "years_field_experience",
            "education",
            "industry",
            "gender",
            # "job",
            "cluster",
            "salary_usd",
            "job_salary_mean",
            "job_salary_std",
            "senior",
            "principal",
            "staff",
            "intern",
            "assistant",
        ]
    ).rename(
        columns={
            "salary_usd": "salary",
            "clean_country": "country",
            "Id": "id",
            "years_field_experience": "experience",
        }
    )

    return df


@hydra.main(config_path="../config", config_name="main")
def feature_data(config: DictConfig):

    with Flow("feature_data") as flow:

        df = (
            load_data(config.clean_data.path1)
            .pipe(age_feature)
            .pipe(education_feature)
            .pipe(gender_feature)
            .pipe(job_feature)
            .pipe(industry_feature)
            # .pipe(salary_features)
            .pipe(experience_feature)
            .pipe(country_feature)
            .pipe(clustering_feature)
            .pipe(feature_estimators)
        )

        df = df.pipe(features)

    flow.run()


if __name__ == "__main__":
    feature_data()
