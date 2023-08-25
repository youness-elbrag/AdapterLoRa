import setuptools
import os
import pkg_resources
import sys
import pathlib

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
# parse_requirements() returns generator of pip.req.InstallRequirement objects
# install_reqs =parse_requirements('requirements.txt', session='hack')
# reqs = [str(ir.req) for ir in install_reqs]
with pathlib.Path('requirements.txt').open() as requirements_txt:
    install_requires = [
        str(requirement)
        for requirement
        in pkg_resources.parse_requirements(requirements_txt)
    ]

setuptools.setup(
    name="AdapterLoRa",
    version="1.1.5",
    author="Youness EL BRAG",
    author_email="younsselbrag@gmail.com",
    description="A Tool for adaptation Larger Transfomer-Based model and Quantization built top on libraries LoRa and LoRa-Torch.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/youness-elbrag/AdapterLoRa/",
    packages=setuptools.find_packages(),
    keywords = ['Quantization', 'AdapterLLM', 'PEFT'],   # Keywords that define your package best
    install_requires= install_requires,
    classifiers=[
    'Development Status :: 4 - Beta',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3.7',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.7',
)

