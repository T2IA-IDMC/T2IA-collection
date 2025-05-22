from setuptools import setup, find_packages

setup(
    name="t2ia_collection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
    ],
    author="Matthieu PELINGRE",
    author_email="matth.pelingre@gmail.com",
    description="A Python package for managing postcard collections with detection features",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/T2IA-IDMC/T2IA-collection",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
