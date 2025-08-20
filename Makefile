all:
	cargo fmt --all
	cargo clippy -- -D warnings
	cargo test
	maturin develop --release

release: all
	maturin build --release
	twine upload --skip-existing target/wheels/*.whl
