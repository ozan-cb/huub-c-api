.PHONY: build release test check clean header

build:
	cargo build --release

release: build

check:
	cargo check --release

test:
	cargo test --release

clean:
	cargo clean -p huub-c-api

# Regenerate include/huub.h without a full rebuild.
header:
	cargo build --release
