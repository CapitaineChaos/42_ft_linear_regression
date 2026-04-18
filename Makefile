VENV    := .venv
PYTHON  := $(VENV)/bin/python3
PIP     := $(VENV)/bin/pip
DATA    := data/data.csv

ARGS := $(filter-out predict,$(MAKECMDGOALS))

.PHONY: all venv install train predict clean fclean re

all: install

$(VENV)/bin/activate:
	python3 -m venv $(VENV)

venv: $(VENV)/bin/activate

install: venv
	$(PIP) install --quiet --upgrade pip
	$(PIP) install --quiet -r requirements.txt

train: install
	cd mandatory && ../$(PYTHON) train.py ../$(DATA)

train-plot: install
	cd mandatory && ../$(PYTHON) train.py ../$(DATA) --plot

predict: install
	cd mandatory && ../$(PYTHON) predict.py $(ARGS)

precision: install
	cd mandatory && ../$(PYTHON) precision.py

%:
	@true

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true

fclean: clean
	rm -rf $(VENV)
	rm -f mandatory/theta0 mandatory/theta1

re: fclean all
