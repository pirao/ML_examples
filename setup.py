from setuptools import setup, find_packages

setup(
    name="LabTDFUtils",
    version="0.0.1",
    author="LabTDF",
    packages=(
        find_packages() +
        find_packages(where='/LabTDFUtils') +
        find_packages(where='/LabTDFUtils/dim_reduction')
    )
)