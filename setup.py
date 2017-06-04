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
        lines_without_comments = list(filter(lambda l: not l.startswith('#'), lines))
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
    packages=['machine_learning'],
    package_dir={'machine_learning': ''},
    scripts=['bin/run.py']
    #entry_points={
    #    'console_scripts': [
    #        'machine_learning=machine_learning:main'
    #        ]
    #    }
)
