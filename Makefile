BUILD_DIR ?= build
BUILD_TYPE ?= Debug
PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
CMAKE ?= cmake
CTEST ?= ctest

ENABLE_CUDA ?= OFF
CIBW_BUILD ?= cp312-*
DIST_DIR ?= dist
VERSION ?=

CUDA_MANYLINUX_IMAGE ?= bmmpy-manylinux_2_28-cuda13.0
CMAKE_ARGS ?= -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBMMPY_ENABLE_CUDA=$(ENABLE_CUDA)
CUDA_WHEEL_CMAKE_ARGS ?= -DBMMPY_ENABLE_CUDA=ON -DBMMPY_CUDA_STATIC_RUNTIME=ON -DCMAKE_CUDA_ARCHITECTURES=all-major

.PHONY: help all configure build cli dev test test-py stubs wheel wheel-tools wheel-manylinux wheel-manylinux-cuda clean distclean clean-build bump_ver release

all: build ## Build default targets

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_.-]+:.*## / {printf "%-22s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

configure: ## Configure CMake in $(BUILD_DIR)
	$(CMAKE) -S . -B $(BUILD_DIR) -G Ninja $(CMAKE_ARGS)

build: configure ## Build native targets
	$(CMAKE) --build $(BUILD_DIR)

cli: build ## Run the CLI executable
	./$(BUILD_DIR)/cli

dev: ## Install editable Python package
	CMAKE_ARGS="$(CMAKE_ARGS)" $(PIP) install -e .

test: build ## Run C++ tests
	$(CTEST) --test-dir $(BUILD_DIR) --output-on-failure

test-py: ## Run Python API tests (run 'make dev' first)
	$(PYTHON) -m unittest discover -s tests -p 'python_api_tests.py'

stubs: configure ## Generate Python stubs
	$(CMAKE) --build $(BUILD_DIR) --target bmmpy_stub

wheel: ## Build a local wheel into $(DIST_DIR)
	rm -rf $(DIST_DIR)
	CMAKE_ARGS="$(CMAKE_ARGS)" $(PIP) wheel . -w $(DIST_DIR)

wheel-tools: ## Install cibuildwheel
	$(PIP) install cibuildwheel

wheel-manylinux: wheel-tools ## Build manylinux wheels
	rm -rf $(DIST_DIR)
	CIBW_BUILD="$(CIBW_BUILD)" \
	CIBW_MANYLINUX_X86_64_IMAGE=manylinux2014 \
	$(PYTHON) -m cibuildwheel --platform linux --output-dir $(DIST_DIR)

wheel-manylinux-cuda: wheel-tools ## Build CUDA manylinux wheels
	rm -rf $(DIST_DIR)
	CIBW_BUILD="$(CIBW_BUILD)" \
	CIBW_MANYLINUX_X86_64_IMAGE=$(CUDA_MANYLINUX_IMAGE) \
	CIBW_ENVIRONMENT='CMAKE_ARGS="$(CUDA_WHEEL_CMAKE_ARGS)" CUDACXX=/usr/local/cuda/bin/nvcc' \
	$(PYTHON) -m cibuildwheel --platform linux --output-dir $(DIST_DIR)

clean: ## Remove products from $(BUILD_DIR)
	@if [ -d "$(BUILD_DIR)" ]; then $(CMAKE) --build $(BUILD_DIR) --target clean; fi

distclean: ## Remove build and dist directories
	rm -rf $(BUILD_DIR) $(DIST_DIR)

clean-build: distclean ## Backward-compatible alias

bump_ver: ## Bump project version
	$(PYTHON) scripts/bump_ver.py

release: ## Publish a release, e.g. make release VERSION=0.3.1
	@[ -n "$(VERSION)" ] || (echo "VERSION is required"; exit 1)
	$(PYTHON) scripts/release.py create $(VERSION) --push