all:
	cargo fmt --all
	cargo clippy -- -D warnings
	cargo test
	maturin develop --release
