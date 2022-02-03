from __future__ import absolute_import
from __future__ import print_function
from setuptools import setup, find_packages

setup(
   name='GNN',
   version='1.0',
   description='Graph Neural Network based algorithm for track finding and pattern recognition',
   author='Nisha Lad',
   # packages=['modules', 'utils'],  #same as name
   packages=find_packages('modules', 'utils'),
   package_dir={'': 'utils'},
   include_package_data = True,
#    install_requires=['wheel', 'bar', 'greek'], #external packages as dependencies
)