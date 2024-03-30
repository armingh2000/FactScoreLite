from setuptools import setup, find_packages

setup(
    name="FactScoreLite",
    version="0.1",
    packages=find_packages(),
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="armingh2000",
    author_email="",
    url="",
    license="MIT",
    install_requires=[
        "numpy",
        "nltk",
        "openai",
        "pytest",
    ],
    package_data={
        "FactScoreLite": [
            "data/*"
        ],  # Include all files in the mypackage/data/ directory
    },
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.org/classifiers/
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
