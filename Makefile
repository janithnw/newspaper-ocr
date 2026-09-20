.PHONY: install jupyter-lab

install:
	uv sync

make lab:
	uv run jupyter lab
