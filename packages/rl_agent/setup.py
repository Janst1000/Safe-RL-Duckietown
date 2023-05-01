from setuptools import setup, find_packages

setup(
    name='rl_agent',
    version='1.0.0',
    description='A safety layer package for autonomous vehicles',
    author='Jan Steinmüller',
    author_email='janst1000@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
)
