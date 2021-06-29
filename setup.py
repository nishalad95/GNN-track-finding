from setuptools import setup

setup(
   name='foo',
   version='1.0',
   description='A useful module',
   author='Nisha Lad',
   packages=['modules', 'utils'],  #same as name
#    install_requires=['wheel', 'bar', 'greek'], #external packages as dependencies
)