.PHONY: test
PY ?= $(HOME)/miniconda3/envs/prl/bin/python

test:
	$(PY) -m pytest -q

-include local.mk
