# import pandas as pd
# from pandera import Check, Column, DataFrameSchema
from pytest_steps import test_steps

from src.clean_data import load_data, rename_selection


@test_steps("rename_selection_steps", "no_nas_step")
def test_process_suite(test_step, steps_data):
    if test_step == "rename_selection_steps":
        rename_columns_step(steps_data)
    elif test_step == "no_nas_step":
        no_nas_step(steps_data)
    pass


def rename_columns_step(steps_data):
    steps_data.raw = load_data("../data/raw_data/dataset.tsv")
    steps_data.intermediate = rename_selection(steps_data.raw)

    assert list(steps_data.intermediate.columns) == [
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


def no_nas_step(steps_data):
    assert sum(steps_data.intermediate.isna().sum()) == 0
