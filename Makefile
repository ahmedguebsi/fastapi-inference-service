# FastAPI Inference Service -- development entry points.
#
# `make help` lists every target. `make check` runs exactly what CI runs.

.DEFAULT_GOAL := help
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

UV := $(shell command -v uv 2>/dev/null)
VENV := .venv
PYTHON := $(VENV)/bin/python
SERVICE := $(VENV)/bin/inference-service

IMAGE ?= inference-service
TAG ?= local
PORT ?= 8000
PYTEST_ARGS ?=

# Image size budget, in bytes. Enforced locally and in CI.
SIZE_BUDGET_MB ?= 200

ifeq ($(OS),Windows_NT)
	PYTHON := $(VENV)/Scripts/python.exe
	SERVICE := $(VENV)/Scripts/inference-service.exe
endif

.PHONY: help
help: ## Show this help
	@echo "FastAPI Inference Service"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
.PHONY: install
install: ## Create the venv and install with dev extras
ifdef UV
	uv venv $(VENV)
	uv pip install --python $(PYTHON) -e ".[dev]"
else
	python3 -m venv $(VENV) || python -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"
endif
	@echo "Ready. Run: make run"

$(PYTHON):
	@$(MAKE) install

.PHONY: clean
clean: ## Remove build, cache and coverage artefacts
	rm -rf build dist *.egg-info src/*.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov
	rm -f .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.py[co]' -delete 2>/dev/null || true

.PHONY: distclean
distclean: clean ## Also remove the virtualenv
	rm -rf $(VENV)

# ---------------------------------------------------------------------------
# Quality gates
# ---------------------------------------------------------------------------
.PHONY: fmt
fmt: $(PYTHON) ## Format with ruff
	$(PYTHON) -m ruff format src tests scripts
	$(PYTHON) -m ruff check --fix src tests scripts

.PHONY: fmt-check
fmt-check: $(PYTHON) ## Verify formatting
	$(PYTHON) -m ruff format --check --diff src tests scripts

.PHONY: lint
lint: $(PYTHON) ## Lint with ruff (includes bandit rules)
	$(PYTHON) -m ruff check src tests scripts

.PHONY: typecheck
typecheck: $(PYTHON) ## Type-check with mypy --strict
	$(PYTHON) -m mypy src

.PHONY: test
test: $(PYTHON) ## Run the test suite with coverage
	$(PYTHON) -m pytest --cov --cov-report=term-missing --cov-report=xml $(PYTEST_ARGS)

.PHONY: test-fast
test-fast: $(PYTHON) ## Unit tests only
	$(PYTHON) -m pytest -m "not integration" -q $(PYTEST_ARGS)

.PHONY: check
check: fmt-check lint typecheck test ## Everything CI runs
	@echo ""
	@echo "All checks passed."

# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
.PHONY: run
run: $(PYTHON) ## Run with auto-reload (development)
	$(SERVICE) --reload --port $(PORT)

.PHONY: serve
serve: $(PYTHON) ## Run without reload
	$(SERVICE) --port $(PORT)

.PHONY: check-config
check-config: $(PYTHON) ## Validate configuration and exit
	$(SERVICE) --check-config

.PHONY: train
train: $(PYTHON) ## Retrain the model artifact
	$(PYTHON) scripts/train_model.py

.PHONY: bench
bench: $(PYTHON) ## Local throughput benchmark
	$(PYTHON) scripts/benchmark.py

.PHONY: smoke
smoke: ## Smoke test a running instance
	./scripts/smoke_test.sh http://localhost:$(PORT)

# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------
.PHONY: docker
docker: ## Build the image
	docker build -t $(IMAGE):$(TAG) .
	@docker images $(IMAGE):$(TAG) --format 'Image size: {{.Size}}'

.PHONY: docker-size
docker-size: docker ## Build and enforce the size budget
	@size=$$(docker image inspect $(IMAGE):$(TAG) --format '{{.Size}}' 2>/dev/null); \
	if [ -z "$$size" ] || [ "$$size" -lt 1000000 ]; then \
		size=$$(docker history --no-trunc --format '{{.Size}}' $(IMAGE):$(TAG) | \
			awk '/[0-9]/ {gsub(/[^0-9.]/,"",$$1); s+=$$1} END {print int(s)}'); \
	fi; \
	limit=$$(( $(SIZE_BUDGET_MB) * 1000 * 1000 )); \
	human=$$(awk -v s="$$size" 'BEGIN {printf "%.1f MB", s/1000/1000}'); \
	echo "Image size: $$human (budget $(SIZE_BUDGET_MB) MB)"; \
	if [ "$$size" -gt "$$limit" ]; then echo "FAIL: over budget"; exit 1; fi; \
	echo "OK: within budget"

.PHONY: docker-run
docker-run: docker ## Run the container
	docker run --rm -it -p $(PORT):8000 \
		--stop-timeout 45 \
		-e INFERENCE_PRESTOP_DELAY_S=0 \
		$(IMAGE):$(TAG)

.PHONY: docker-scan
docker-scan: docker ## Scan the image with trivy
	trivy image --severity HIGH,CRITICAL --exit-code 1 $(IMAGE):$(TAG)

.PHONY: demo
demo: ## Start the compose stack
	docker compose up --build -d
	@echo ""
	@echo "  API         http://localhost:8000/docs"
	@echo "  Prometheus  http://localhost:9090"
	@echo "  Grafana     http://localhost:3000"
	@echo ""
	@echo "  Stop with: make demo-down"

.PHONY: demo-down
demo-down: ## Stop the compose stack
	docker compose down -v

.PHONY: demo-logs
demo-logs: ## Follow the API logs
	docker compose logs -f api

# ---------------------------------------------------------------------------
# Release
# ---------------------------------------------------------------------------
.PHONY: build
build: $(PYTHON) ## Build sdist and wheel
	$(PYTHON) -m pip install --quiet build
	$(PYTHON) -m build

.PHONY: openapi
openapi: $(PYTHON) ## Export the OpenAPI schema
	$(PYTHON) -c "import json; from inference_service.main import create_app; \
		print(json.dumps(create_app().openapi(), indent=2))" > openapi.json
	@echo "wrote openapi.json"
