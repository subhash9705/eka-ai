"""
setup.py — Legacy build script for editable installs and older tooling.

Modern installs use pyproject.toml. This file exists for compatibility with
tools that still call `python setup.py develop / install`.
"""

from setuptools import setup

if __name__ == "__main__":
    setup()
