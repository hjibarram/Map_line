#!/usr/bin/env python
import os
from setuptools import setup, find_packages

def version(fn):
    v = '1.0.1'
    with open(fn, 'r') as f:
        for l in f.readlines():
            if '__version__' in l:
                v = l.split('=')[-1].strip().replace("'", '').split(' ')[-1][1:]
    return v

#def readme():
#   with open('README.md') as f:
#       return f.read()

requirements = [
    'astropy',#>=3.2.1',
    'matplotlib',#>=3.1.1',
    'numpy',#>=1.17.2',
    'scipy',#>=1.3.0',
    'cloup',
    'click',
    'emcee',
    'tqdm',
]

DATA_DIRNAME = 'data'
SCRIPTS_DIRNAME = 'bin'
VERSION_FILE = 'MapLines/common/constants.py'

all_packages = find_packages()
packages_data = {
    package: [f'{DATA_DIRNAME}/*']+[f'{os.path.join(DATA_DIRNAME, sub)}/*' for root, subs, files in os.walk(os.path.join(package, DATA_DIRNAME)) for sub in subs]
    for package in all_packages if os.path.isdir(os.path.join(package, DATA_DIRNAME))
}
scripts = ["bin/run_mapline"]
#    os.path.join(SCRIPTS_DIRNAME, script_name)
#    for script_name in os.listdir(SCRIPTS_DIRNAME) if script_name.endswith('.py')
#]
version = version(VERSION_FILE)

setup(
    name='MapLine',
    version=version,
    description='A Python implementation of the 3D emission line fitting code',
    #long_description=readme(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    keywords='galaxies',
    url='https://github.com/hjibarram/Map_line',
    download_url=f'https://github.com/hjibarram/Map_line/archive/refs/tags/v{version}.tar.gz',
    author='hjibarram',
    author_email='hjibarram@gmail.com',
    license='MIT',
    packages=all_packages,
    setup_requires=['wheel'],
    install_requires=requirements,
    include_package_data=True,
    package_data=packages_data,
    scripts=scripts,
    zip_safe=False,
    test_suite='nose.collector',
    tests_require=['nose'],
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
