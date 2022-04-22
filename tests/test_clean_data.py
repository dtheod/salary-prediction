import numpy as np
import pandas as pd
from pandera import Check, Column, DataFrameSchema
from pytest_steps import test_steps

from src.clean_data import currency_conversion, rename_selection


@test_steps(
    "rename_selection_step",
    "check_nas_step",
    "conversion_step",
    "conversion_data_check_step",
)
def test_process_suite(test_step, steps_data):
    if test_step == "rename_selection_step":
        rename_columns_step(steps_data)
    elif test_step == "check_nas_step":
        check_nas_step(steps_data)
    elif test_step == "conversion_step":
        conversion_step(steps_data)
    elif test_step == "conversion_data_check_step":
        conversion_data_check_step(steps_data)


def rename_columns_step(steps_data):

    data = pd.DataFrame(
        {
            "Timestamp": ["4/27/2021 11:02:10"],
            "How old are you?": ["25-34"],
            "Industry": ["Education (Higher Education)"],
            "Job title": ["Research and Instruction Librarian"],
            "Additional context on job title": [np.nan],
            "Annual salary": ["55,000"],
            "Other monetary comp": 0.0,
            "Currency": ["GBP"],
            "Currency - other": [np.nan],
            "Additional context on income": [np.nan],
            "Country": ["United Kingdom"],
            "State": ["Massachusetts"],
            "City": ["Boston"],
            "Overall years of professional experience": ["5-7 years"],
            "Years of experience in field": ["5-7 years"],
            "Highest level of education completed": ["Master's degree"],
            "Gender": ["Woman"],
            "Race": ["White"],
        }
    )

    renamed_data = rename_selection.run(data)
    assert list(renamed_data.columns) == [
        "Id",
        "age",
        "Industry",
        "job",
        "salary",
        "currency",
        "country",
        "years_field_experience",
        "education",
        "gender",
    ]
    steps_data.intermediate_a = renamed_data


def check_nas_step(steps_data):
    assert sum(steps_data.intermediate_a.isna().sum()) == 0


def conversion_step(steps_data):
    intermediate_b = currency_conversion.run(steps_data.intermediate_a)
    assert intermediate_b["salary_usd"][0] >= 70800

    steps_data.intermediate_b = intermediate_b


def conversion_data_check_step(steps_data):

    schema = DataFrameSchema(
        {
            "Id": Column(int),
            "age": Column(object),
            "Industry": Column(object),
            "job": Column(object),
            "salary": Column(int, Check.greater_than_or_equal_to(0)),
            "currency": Column(object),
            "country": Column(object),
            "years_field_experience": Column(object),
            "education": Column(object),
            "gender": Column(object),
            "salary_usd": Column(float, Check.greater_than_or_equal_to(0)),
            "new_currency": Column(object),
        }
    )

    schema.validate(steps_data.intermediate_b)
