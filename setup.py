#!/usr/bin/env python

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

from os import path

def get_requirements():
    """Reads the installation requirements from requirements.pip"""
    with open("requirements.pip") as f:
        lines = f.read().split("\n")
        lines_without_comments = filter(lambda l: not l.startswith('#'), lines)
        return lines_without_comments

here = path.abspath(path.dirname(__file__))

setup(name='machine_learning',
    description='Machine Learning Project',
    long_description='',
    author='Diane Losardo',
    url='todo',
    download_url='todo',
    author_email='dlosardo@gmail.com',
    version='0.1',
    install_requires=get_requirements(),
    packages=find_packages(),
    scripts=['bin/run.py'],
    entry_points={
        'console_scripts': [
            'machine_learning=machine_learning:main'
            ]
        }
)
