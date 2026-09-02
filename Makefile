.PHONY: test verify demo k25

test:
	python run_all.py

verify:
	python scripts/verify_repository.py

demo:
	python -m demos.demo_core
	python -m demos.demo_temporal
	python -m demos.demo_vision
	python -m demos.demo_monolith

k25:
	python research/vision/k25_multi_sensor_selection.py
