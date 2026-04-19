VENV    := .venv
PYTHON  := $(VENV)/bin/python3
PIP     := $(VENV)/bin/pip
DATA    := data/data.csv

MANDATORY_DIR := mandatory
BONUS_DIR     := bonus

ARGS   := $(filter-out predict,$(MAKECMDGOALS))
_NARG  := $(filter-out test-bonus-weird test-weird,$(MAKECMDGOALS))

.PHONY: all venv install train predict clean fclean re test-weird test-bonus-weird

all: install

$(VENV)/bin/activate:
	python3 -m venv $(VENV)

venv: $(VENV)/bin/activate

install: venv
	$(PIP) install --quiet --upgrade pip
	$(PIP) install --quiet -r requirements.txt

# oxxxxxxx[======== Mandatory part ==========>

train: install
	@cd $(MANDATORY_DIR) && ../$(PYTHON) train.py ../$(DATA) || true

predict: install
	@cd $(MANDATORY_DIR) && ../$(PYTHON) predict.py $(ARGS) || true

# oxxxxxxx[======== Bonus part ==============>

train-bonus: install
	@cd $(BONUS_DIR) && ../$(PYTHON) train.py ../$(DATA) || true

test-bonus-weird: install
	@cd $(BONUS_DIR) && ../$(PYTHON) train.py ../data/weird_$(_NARG).csv || true

test-weird: install
	@cd $(MANDATORY_DIR) && ../$(PYTHON) train.py ../data/weird_$(_NARG).csv || true

predict-bonus: install
	@cd $(BONUS_DIR) && ../$(PYTHON) predict.py $(ARGS) || true

precision-bonus: install
	@cd $(BONUS_DIR) && ../$(PYTHON) precision.py ../$(DATA) || true

# ==============================================================================

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