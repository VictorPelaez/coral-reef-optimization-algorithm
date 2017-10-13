#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup
import os

# python setup.py sdist
# python setup.py bdist_wheel --universal
# cd C:\Users\victor\Anaconda3\Scripts
# twine upload C:/Users/victor/Documents/Repositorios/coral-reef-optimization-algorithm/dist/*
# pip install cro --upgrade --no-cache-dir

with open('README.txt') as file:
    long_description = file.read()

setup(
    name='cro',
    version='0.0.4.10',
    author='Victor Pelaez',
    author_email='victor.m.pelaez@outlook.com',
    packages= ['cro'],
    url='https://github.com/VictorPelaez/coral-reef-optimization-algorithm',
    license = 'MIT',
    description='Coral Reef Optimization (CRO) Algorithm',
    long_description= long_description,
    keywords ='optimization algorithm meta-heuristic coral reef',
    #python_requires='>=3',
    classifiers = [],

)

