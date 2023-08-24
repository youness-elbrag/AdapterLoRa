import setuptools
import os
import sys

current_dir = os.path.dirname(__file__)
target_dir = os.path.abspath(os.path.join(current_dir, "."))
sys.path.insert(0, target_dir)

with open(target_dir +"/README.md", "r") as fh:
    long_description = fh.read()

"""
Setup AdapterLoRa .

ENV:
    python -m venv AdapterLoRa && pip install -r requirements.txt    
"""

setuptools.setup(
    name="AdapterLoRa",
    version="1.1.0",
    author="Youness EL BRAG",
    author_email="younsselbrag@gmail.com",
    description="A Tool for adaptation Larger Transfomer-Based model and Quantization built top on libraries LoRa and LoRa-Torch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/youness-elbrag/AdapterLoRa/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)