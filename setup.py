from setuptools import setup, find_packages

setup(
    name="phd_project",
    version="0.1",
    packages=find_packages(include=['utils', 'env']),
    install_requires=[
        'pandas',
    ],
) 