from __future__ import absolute_import
from __future__ import print_function
from setuptools import setup, find_packages

setup(
   name='GNN',
   version='1.0',
   description='Graph Neural Network based algorithm for track finding and pattern recognition',
   author='Nisha Lad',
   packages=find_packages('src'),
   package_dir={'': 'src'},
   include_package_data = True,
#    install_requires=['networkx', 'filterpy', 'python-louvain', 'scipy', 'numpy', 'pandas', 'glob'], #external packages as dependencies
)