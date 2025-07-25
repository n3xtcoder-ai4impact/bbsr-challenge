from setuptools import setup, find_packages

setup(
    name="pollutant_predictor",
    version="0.1.0",
    description="A model to predict environmental pollutants in construction materials",
    author="Your Name",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pandas>=1.0",
        "scikit-learn>=1.0",
        # list any other dependencies here
    ],
)

