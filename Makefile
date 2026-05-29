# Build shim for huub-c-api. Self-contained — does not require any
# specific external env. Mirrors tools/huub_eval/Makefile's pattern but
# pins to Rust 1.88 (the verified-minimum toolchain from
# PRODUCTIZATION.md §1; huub's `bon` transitive dep bumped MSRV past
# 1.86, so the workspace SDK toolchain doesn't build).
#
# Locations:
#   - rustup 1.88 toolchain: ~/.rustup_188 (NFS, visible on compute nodes)
#   - CARGO_HOME pinned to the vendored huub crate's cargo-home so we
#     reuse its registry cache (~/.cargo is intentionally broken)
#   - own target/ dir (don't share with huub_eval which uses a different
#     toolchain)

WORKSPACE := /net/ozan-vm/srv/nfs/ozan-data/ws/workspace/csp
HUUB_DIR  := $(WORKSPACE)/third_party/huub
CARGO_HOME_DIR := $(HUUB_DIR)/.cargo-home

RUSTUP_HOME_DIR := $$HOME/.rustup_188
RUST_BIN        := $(RUSTUP_HOME_DIR)/toolchains/1.88.0-x86_64-unknown-linux-gnu/bin

CBRUN_ENV := \
  export PATH=$(RUST_BIN):$$PATH ; \
  export RUSTUP_HOME=$(RUSTUP_HOME_DIR) ; \
  export CARGO_HOME=$(CARGO_HOME_DIR) ; \
  cd $(WORKSPACE)/tools/huub-c-api

.PHONY: build release test check clean header

build:
	cbrun -t rocky -- srun -c 4 --mem=8G bash -lc '$(CBRUN_ENV) ; cargo build --release'

release: build

check:
	cbrun -t rocky -- srun -c 4 --mem=8G bash -lc '$(CBRUN_ENV) ; cargo check --release'

test:
	cbrun -t rocky -- srun -c 4 --mem=8G bash -lc '$(CBRUN_ENV) ; cargo test --release'

clean:
	cbrun -t rocky -- srun -c 2 bash -lc '$(CBRUN_ENV) ; cargo clean -p huub-c-api'

# Regenerate include/huub.h without a full rebuild (still goes through
# build.rs / cbindgen).
header: build
