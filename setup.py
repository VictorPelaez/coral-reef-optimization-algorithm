#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

# python ../Documents/Repositorios/coral-reef-optimization-algorithm/setup.py sdist
# python ../Documents/Repositorios/coral-reef-optimization-algorithm/setup.py sdist upload -r https://python.org/pypi

setup(
    name='cro',
    version='0.1dev',
    author='Victor Pelaez',
    author_email='victor.m.pelaez@outlook.com',
    package_dir={'': '.'},
    packages=find_packages('.'),
    url='https://github.com/VictorPelaez/coral-reef-optimization-algorithm',
    license='LICENSE.txt',
    description='Coral Reef Optimization (CRO) Algorithm',
    #long_description = open('README.txt').read(),    
    keywords='optimization algorithm meta-heuristic coral reef',
    python_requires='>=3'
    #install_requires=["numpy","pandas"]
)