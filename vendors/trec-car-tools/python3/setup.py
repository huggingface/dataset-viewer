#!/usr/bin/env python3
from setuptools import setup

setup(
    name='trec-car-tools',
    version='2.5.4',
    packages=['trec_car'],
    url='https://github.com/TREMA-UNH/trec-car-tools/python3',
    # download_url='https://github.com/TREMA-UNH/trec-car-tools/archive/2.0.tar.gz',
    keywords=['wikipedia','complex answer retrieval','trec car'],
    license='BSD 3-Clause',
    author='laura-dietz',
    author_email='Laura.Dietz@unh.edu',
    description='Support tools for TREC CAR participants. Also see trec-car.cs.unh.edu',
    install_requires=['cbor>=1.0.0', 'numpy>=1.11.2'],
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
       ]
)
