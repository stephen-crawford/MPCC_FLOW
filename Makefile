# mpcc_flow — minimal developer targets
.PHONY: test test-cov test-integration test-e2e venv lint

PYTHON ?= python3
VENV := .venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest

venv:
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)
	$(PIP) install -U pip setuptools wheel
	$(PIP) install -e ".[dev]"

test:
	@test -f $(PYTEST) || { echo "Run: make venv"; exit 1; }
	$(PYTEST) tests/ -v --tb=short

test-cov:
	@test -f $(PYTEST) || { echo "Run: make venv"; exit 1; }
	$(PYTEST) tests/ \
		--cov=pympc --cov=planning --cov=modules --cov=solver \
		--cov=network_emulation --cov=utils \
		--cov-report=term-missing --tb=short

test-integration:
	@test -f $(PYTEST) || { echo "Run: make venv"; exit 1; }
	$(PYTEST) tests/test_mahimahi_nimbus_integration.py -v --tb=short

test-e2e:
	@test -f $(PYTEST) || { echo "Run: make venv"; exit 1; }
	$(PYTEST) tests/test_mahimahi_nimbus_integration.py tests/test_network_stack_e2e.py -v --tb=short

lint:
	@test -f $(VENV)/bin/ruff || { echo "Run: make venv"; exit 1; }
	$(VENV)/bin/ruff check pympc planning modules solver utils network_emulation tests
