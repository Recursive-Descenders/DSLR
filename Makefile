# This is a copy of my linear regretion Makefile, change as we develop the project

# Extra words after the real target (e.g. `make predict dataset_test.csv model/model.json`)
# are otherwise separate goals and Make will treat existing files as "up to date" / "nothing to do".
# Mark them phony so they only satisfy the goal list without touching the real files.
ifneq ($(word 2,$(MAKECMDGOALS)),)
  EXTRA_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  .PHONY: $(EXTRA_ARGS)
  $(EXTRA_ARGS):
	@:
endif

VENV := .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

.PHONY: default setup describe train predict evaluate confusion clean clean-all help
default: setup

setup: $(VENV)/bin/python
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

$(VENV)/bin/python:
	python3 -m venv $(VENV)

histogram:
	@$(PY) src/histogram.py $(filter-out $@,$(MAKECMDGOALS))
  
describe:
	@$(PY) src/describe.py $(filter-out $@,$(MAKECMDGOALS))

scatter_plot:
	@$(PY) src/scatter_plot.py $(filter-out $@,$(MAKECMDGOALS))

train:
	@$(PY) src/logreg_train.py $(filter-out $@,$(MAKECMDGOALS))

predict:
	$(PY) src/logreg_predict.py $(filter-out $@,$(MAKECMDGOALS))

evaluate:
	@$(PY) src/logreg_evaluate.py $(filter-out $@,$(MAKECMDGOALS))

confusion:
	@$(PY) src/confusion_matrix.py $(filter-out $@,$(MAKECMDGOALS))

clean:
	rm -rf __pycache__ src/__pycache__ model/*.json plots/*.png plot.png epochs.gif

clean-all: clean
	rm -rf $(VENV)
