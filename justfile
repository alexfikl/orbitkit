PYTHON := 'python -X dev'

_default:
    @just --list

# {{{ formatting

alias fmt: format

[doc('Reformat all source code')]
format: isort black pyproject justfmt

[doc('Run ruff isort fixes over the source code')]
isort:
    ruff check --fix --select=I src
    ruff check --fix --select=RUF022 src
    @echo -e "\e[1;32mruff isort clean!\e[0m"

[doc('Run ruff format over the source code')]
black:
    ruff format src
    @echo -e "\e[1;32mruff format clean!\e[0m"

[doc('Run pyproject-fmt over the configuration')]
pyproject:
    {{ PYTHON }} -m pyproject_fmt --indent 4 pyproject.toml
    @echo -e "\e[1;32mpyproject clean!\e[0m"

[doc('Run just --fmt over the justfile')]
justfmt:
    just --unstable --fmt
    @echo -e "\e[1;32mjust --fmt clean!\e[0m"

# }}}
# {{{ linting

[doc('Run all linting checks over the source code')]
lint: typos reuse ruff mypy

[doc('Run typos over the source code and documentation')]
typos:
    typos --sort
    @echo -e "\e[1;32mtypos clean!\e[0m"

[doc('Check REUSE license compliance')]
reuse:
    {{ PYTHON }} -m reuse lint
    @echo -e "\e[1;32mREUSE compliant!\e[0m"

[doc('Run ruff checks over the source code')]
ruff:
    ruff check
    @echo -e "\e[1;32mruff clean!\e[0m"

[doc('Run mypy checks over the source code')]
mypy:
    {{ PYTHON }} -m mypy src tests
    @echo -e "\e[1;32mmypy clean!\e[0m"

# }}}
# {{{ pin

[private]
requirements_build_txt:
    uv pip compile --upgrade --universal --python-version "3.10" \
        -o .ci/requirements-build.txt .ci/requirements-build.in

[private]
requirements_test_txt:
    uv pip compile --upgrade --universal --python-version '3.10' \
        --extra test \
        -o .ci/requirements-test.txt pyproject.toml

[private]
requirements_txt:
    uv pip compile --upgrade --universal --python-version '3.10' \
        -o requirements.txt pyproject.toml

[doc('Pin dependency versions to requirements.txt')]
pin: requirements_txt requirements_build_txt requirements_test_txt

# }}}
# {{{ develop

[doc('Install project in editable mode')]
develop:
    @rm -rf build
    @rm -rf dist
    {{ PYTHON }} -m pip install \
        --verbose \
        --no-build-isolation \
        --editable .

[doc("Editable install using pinned dependencies from requirements-test.txt")]
pip-install:
    {{ PYTHON }} -m pip install --verbose --requirement .ci/requirements-build.txt
    {{ PYTHON }} -m pip install \
        --verbose \
        --requirement .ci/requirements-test.txt \
        --no-build-isolation \
        --editable .

[doc("Remove various build artifacts")]
clean:
    rm -rf build dist
    rm -rf docs/_build

[doc("Remove various temporary files")]
purge: clean
    rm -rf .ruff_cache .pytest_cache .mypy_cache tags

[doc("Regenerate ctags")]
ctags:
    ctags --recurse=yes \
        --tag-relative=yes \
        --exclude=.git \
        --exclude=docs \
        --python-kinds=-i \
        --language-force=python

# }}}
# {{{ tests

[doc("Run pytest tests")]
test *PYTEST_ADDOPTS:
    {{ PYTHON }} -m pytest {{ PYTEST_ADDOPTS }}

# }}}
