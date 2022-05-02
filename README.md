# salary-prediction

[![Run tests](https://github.com/dtheod/salary-prediction/actions/workflows/run_tests.yml/badge.svg?branch=main)](https://github.com/dtheod/salary-prediction/actions/workflows/run_tests.yml)

End-to-End Salary Prediction

## Project Structure
* `src`: consists of Python scripts
* `config`: consists of configuration files
* `data`: consists of data
* `notebook`: consists of Jupyter Notebooks
* `tests`: consists of test files

## Set Up the Project
1. Install [Poetry](https://python-poetry.org/docs/#installation)
2. Set up the environment:
```bash
make setup
make install_all
```
3. To persist the output of Prefect's flow, run 
```bash
export PREFECT__FLOWS__CHECKPOINTING=true
```

## Run the Project
To run all flows, type:
```bash
python src/main.py
```

## Run the API Endpoint
To run type:
```bash
bentoml serve src/bentoml_run.py:service --reload
```
## Run the Streamlit App
```bash
streamlit run src/streamlit_app.py  
```


