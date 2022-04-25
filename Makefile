.PHONY: notebook docs
.EXPORT_ALL_VARIABLES:

PREFECT__FLOWS__CHECKPOINTING=true

install: 
	@echo "Installing..."
	poetry install

activate:
	@echo "Activating virtual environment"
	poetry shell

env:
	@echo "Please set the environment variable 'PREFECT__FLOWS__CHECKPOINTING=true' to persist the output of Prefect's flow"

pull_data:
	@echo "Pulling data..."
	poetry run dvc pull


docs_view:
	@echo View API documentation
	pdoc src

docs_save:
	@echo Save documentation to docs... 
	pdoc src -o docs

setup: activate 
install_all: install env

test:
	pytest --no-header -v  

clean:
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	rm -rf .pytest_cache
	find . -name "*__pycache__" -exec rm -rf {} \;
	find . -name "*.pytest_cache" -exec rm -rf {} \;
	find . -type d -name ".DS_Store" -delete