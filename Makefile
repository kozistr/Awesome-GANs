.PHONY: init check format requirements

init:
	pip3 install -U pipenv
	pipenv install --dev

check:
	isort --recursive --check-only awesome_gans
	black -S -l 120 --check awesome_gans
	pylint awesome_gans

format:
	isort -rc -y awesome_gans
	black -S -l 120 awesome_gans

requirements:
	pipenv lock -r > requirements.txt
