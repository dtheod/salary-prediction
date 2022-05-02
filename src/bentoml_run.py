import bentoml
import bentoml.sklearn
import numpy as np
import pandas as pd
from bentoml.io import JSON, NumpyNdarray
from pydantic import BaseModel

scaler = bentoml.sklearn.load_runner("scaler:latest", function_name="transform")

encoder = bentoml.sklearn.load_runner("one_hot:latest", function_name="transform")

cluster = bentoml.sklearn.load_runner("kmeans", function_name="predict")

tfidf = bentoml.sklearn.load_runner("tfidf:latest", function_name="transform")

regressor = bentoml.xgboost.load_runner("booster_tree_df:latest")

service = bentoml.Service(
    "prediction", runners=[scaler, encoder, cluster, tfidf, regressor]
)

# df = pd.DataFrame({'age':['25-34'],
#                    'Industry':['Education (Higher Education)'],
#                    'job': ['senior program manager'],
#                    'country': ['united states'],
#                    'years_field_experience':['5-7 years'],
#                    'education':['College degree'],
#                    'gender': ['Man']
# })


class Employee(BaseModel):
    age: str = "25-34"
    industry: str = "Education (Higher Education)"
    job: str = "program manager"
    country: str = "united states"
    years_field_experience: str = "6"
    education: str = "14"
    gender: str = "Man"
    senior: str = "1"
    principal: str = "0"
    staff: str = "0"
    assistant: str = "0"
    intern: str = "0"


@service.api(input=JSON(pydantic_model=Employee), output=NumpyNdarray())
def predict(df: Employee) -> np.ndarray:

    df1 = pd.DataFrame(df.dict(), index=[0]).pipe(
        lambda df_: df_.assign(
            education=pd.to_numeric(df_.education),
            years_field_experience=pd.to_numeric(df_.years_field_experience),
            senior=pd.to_numeric(df_.senior),
            principal=pd.to_numeric(df_.principal),
            staff=pd.to_numeric(df_.staff),
            assistant=pd.to_numeric(df_.assistant),
            intern=pd.to_numeric(df_.intern),
        )
    )

    res = tfidf.run(df1["job"][0])

    df2 = (
        df1.assign(cluster="c" + str(cluster.run(res.toarray().flatten())))
        .drop("job", axis=1)
        .select_dtypes("object")
        .apply(encoder.run, axis=1)
    )

    rest = df1.select_dtypes(exclude="object")
    features_ready = pd.concat(
        [pd.DataFrame({"id": [1]}), rest, pd.DataFrame(list(df2))], axis=1
    )
    features_scaled_ready = scaler.run(features_ready).reshape(1, -1)
    result = regressor.run(pd.DataFrame(features_scaled_ready))
    return result
