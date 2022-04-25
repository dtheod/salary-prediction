# from cgi import test
# from pytest_steps import test_steps
# from src.outlier_removal import outlier_detect_remove, load_data


# @test_steps("outlier_removal_step")
# def test_process_suite(test_step, steps_data):
#     if test_step == "outlier_removal_step":
#         return test_outlier_remove_step(steps_data)


# def test_outlier_remove_step(steps_data):
#     steps_data.raw = load_data("../data/clean_data/currency_conversion.csv")
#     steps_data.intermediate = outlier_detect_remove(steps_data.raw)
#     assert steps_data.intermediate['salary_usd'].min() > 1000
#     assert steps_data.intermediate['salary_usd'].max() < 1e7
