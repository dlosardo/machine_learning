#!/usr/bin/env python3

#try:
#    from setuptools import setup, find_packages
#except ImportError:
from setuptools import setup
#from distutils.core import setup

from os import path

def get_requirements():
    """Reads the installation requirements from requirements.pip"""
    with open("requirements.pip") as f:
        lines = f.read().split("\n")
        requirements = list(filter(lambda l: not l.startswith('#'), lines))
        return requirements

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
    packages=['machine_learning', 'machine_learning/model_utils'
              , 'machine_learning/utils', 'machine_learning/algorithm'
              , 'machine_learning/driver', 'machine_learning/cost_function'
              , 'machine_learning/hypothesis'],
    package_dir={'machine-learning': ''},
    scripts=['bin/run.py']
)
