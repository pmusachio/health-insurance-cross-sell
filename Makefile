.PHONY: install profile train analyze test api

install:
	python -m pip install -r requirements.txt

profile:
	PYTHONPATH=src python -m health_insurance_cross_sell.cli profile

train:
	PYTHONPATH=src python -m health_insurance_cross_sell.cli train

analyze:
	PYTHONPATH=src python -m health_insurance_cross_sell.cli analyze

test:
	python -m pytest

api:
	PYTHONPATH=src uvicorn health_insurance_cross_sell.api:app --reload
