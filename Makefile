# FreshRoute (ripple). Override the interpreter with: make PY=path/to/python reproduce
PY ?= python
export PYTHONPATH := src

.PHONY: install data check reproduce test app clean-results

install:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m freshroute.adapters.synthetic

check:
	$(PY) -m freshroute.adapters.check

reproduce:
	$(PY) -m freshroute.reproduce

test:
	$(PY) -m pytest -q

app:
	$(PY) -m streamlit run app/Home.py

clean-results:
	$(PY) -c "import shutil; shutil.rmtree('results', ignore_errors=True)"
