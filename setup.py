from setuptools import setup, find_packages

setup(
    name="ma_darts",
    version="0.1.0",
    description="Master Thesis darts scoring system",
    author="JuuuF",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11.0",
    install_requires=[
        # TODO
    ],
)
