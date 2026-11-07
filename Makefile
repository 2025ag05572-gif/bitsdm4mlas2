PY      ?= .venv/bin/python
CONFIG  ?= configs/config.yaml
PREFECT_PORT ?= 4210
export PREFECT_HOME := $(CURDIR)/.prefect

.PHONY: setup sample pipeline flow prefect-server mlflow-ui test recommend report clean

setup:            ## create the virtualenv and install pinned dependencies
	python3.11 -m venv .venv && $(PY) -m pip install --upgrade pip && $(PY) -m pip install -r requirements.txt

sample:           ## run the whole pipeline on a small synthetic Food.com-format sample
	$(PY) -m scripts.make_sample_data && $(PY) -m src.pipeline --config configs/config.sample.yaml

pipeline:         ## run all stages without an orchestrator
	$(PY) -m src.pipeline --config $(CONFIG)

prefect-server:   ## local Prefect UI on http://127.0.0.1:$(PREFECT_PORT) (loopback only)
	.venv/bin/prefect server start --host 127.0.0.1 --port $(PREFECT_PORT)

flow:             ## run the Prefect flow against the local server
	PREFECT_API_URL=http://127.0.0.1:$(PREFECT_PORT)/api $(PY) dags/recommendation_pipeline.py --config $(CONFIG)

mlflow-ui:        ## MLflow UI on http://127.0.0.1:5000 (loopback only)
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

test:
	$(PY) -m pytest -q tests

recommend:        ## e.g. make recommend USER=424680
	$(PY) -m src.serving.recommend --user-id $(USER) --model hybrid --k 10

report:           ## render reports/final_report.md to HTML/PDF
	$(PY) -m scripts.build_report

clean:
	rm -rf data/raw/*.parquet data/processed/* data/features/* data/errors/*.parquet models/*.joblib
