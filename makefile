PATH_TO_PYPY="/usr/bin/pypy3"

install: venv
	. venv/bin/activate; pip3 install -Ur requirements.txt

venv :
	test -d venv || python3 -m venv venv

profile: venv
	. venv/bin/activate; python -m vmprof ./src/main.py

compile:
	. venv/bin/activate; python setup.py build_ext --inplace

clean:
	rm -rf venv
	find -iname "*.pyc" -delete