VENV    := .venv
PYTHON  := $(VENV)/bin/python3
PIP     := $(VENV)/bin/pip
DATA    := data/data.csv

MANDATORY_DIR := mandatory
BONUS_DIR     := bonus

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
	@cd $(MANDATORY_DIR) && ../$(PYTHON) train.py ../$(DATA) || true

train-plot: install
	@cd $(BONUS_DIR) && ../$(PYTHON) train.py ../$(DATA) --plot || true

predict: install
	@cd $(MANDATORY_DIR) && ../$(PYTHON) predict.py $(ARGS) || true

precision: install
	@cd $(BONUS_DIR) && ../$(PYTHON) precision.py ../$(DATA) || true

%:
	@true

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true

fclean: clean
	rm -rf $(VENV)
	rm -f $(MANDATORY_DIR)/theta0 $(MANDATORY_DIR)/theta1
	rm -f $(BONUS_DIR)/theta0 $(BONUS_DIR)/theta1

re: fclean all

git-fix:
	@bash scripts/git-fix-remote.sh