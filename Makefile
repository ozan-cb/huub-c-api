# Build shim for huub-c-api. Mirrors tools/huub_eval/Makefile:
# - rustup 1.95 from ~/.rustup (NFS, visible on compute nodes)
# - CARGO_HOME pinned to a workspace-local dir (~/.cargo is intentionally
#   a broken symlink)
# - CARGO_TARGET_DIR shared with the huub build to avoid recompiling
#   huub deps

WORKSPACE := /net/ozan-vm/srv/nfs/ozan-data/ws/workspace/csp
HUUB_DIR  := $(WORKSPACE)/third_party/huub
CARGO_HOME_DIR := $(HUUB_DIR)/.cargo-home
# Use a dedicated target dir so the 1.88 build artifacts don't collide
# with the 1.95 huub_eval target dir.
TARGET_DIR     := $(WORKSPACE)/tools/huub-c-api/target

# Rust 1.88 — the minimum version that satisfies the `bon`/`darling`
# transitive MSRV bumps (see ../../PRODUCTIZATION.md §1). This is the
# toolchain we'll ship against in production once the SDK Rust is
# upgraded (or once we patch `bon` down to 1.86-compatible).
RUST_BIN := $$HOME/.rustup_188/toolchains/1.88.0-x86_64-unknown-linux-gnu/bin

CBRUN_ENV := \
  export PATH=$(RUST_BIN):$$PATH RUSTUP_HOME=$$HOME/.rustup_188 ; \
  export CARGO_HOME=$(CARGO_HOME_DIR) ; \
  export CARGO_TARGET_DIR=$(TARGET_DIR) ; \
  cd $(WORKSPACE)/tools/huub-c-api

.PHONY: build release test check clean header

build:
	cbrun -t rocky -- srun -c 8 bash -lc '$(CBRUN_ENV) ; cargo build --release'

release: build

check:
	cbrun -t rocky -- srun -c 4 bash -lc '$(CBRUN_ENV) ; cargo check --release'

test:
	cbrun -t rocky -- srun -c 8 bash -lc '$(CBRUN_ENV) ; cargo test --release'

clean:
	cbrun -t rocky -- srun -c 2 bash -lc '$(CBRUN_ENV) ; cargo clean -p huub-c-api'

# Regenerate the C header without rebuilding the libraries. The build.rs
# always tries to regenerate, but this is a fast path for header-only
# changes.
header:
	cbrun -t rocky -- srun -c 2 bash -lc '$(CBRUN_ENV) ; cargo build --release --offline 2>/dev/null || cargo build --release'
