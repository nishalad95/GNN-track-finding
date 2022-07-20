from __future__ import absolute_import
from __future__ import print_function
from setuptools import setup, find_packages
import distutils.command.build


setup(
   name='learn_KL_parabolic_model',
   version='1.0',
   author='Nisha Lad',
   packages=find_packages('src'),
   package_dir={'': 'src'},
   include_package_data = True,
)