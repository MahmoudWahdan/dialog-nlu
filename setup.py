"""
Originally guided by AllenNLP and Transformers repos
To create the package for pypi.
"""

import shutil
from pathlib import Path

from setuptools import find_packages, setup


# Remove stale dialognlu.egg-info directory to avoid https://github.com/pypa/pip/issues/5466
stale_egg_info = Path(__file__).parent / "dialognlu.egg-info"
if stale_egg_info.exists():
    print("Warning: {} exists.".format(stale_egg_info))
    shutil.rmtree(stale_egg_info)


extras = {}

setup(
    name="dialognlu",
    version="0.2.0",
    author="Mahmoud Wahdan",
    author_email="mahmoud.a.wahdan@gmail.com",
    description="State-of-the-art Dialog NLU (Natural Language Understanding) Library with TensorFlow 2.x and keras",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="NLP deep learning transformers BERT ALBERT tensorflow",
    license="Apache",
    url="https://github.com/MahmoudWahdan/dialog-nlu",
    
    # package_dir={"dialognlu": "dialognlu"},
    # packages=find_packages(),
    
    package_dir={"": "src"},
    packages=find_packages("src"),
    
    install_requires=[
        "scikit-learn>=0.23.2", "pandas>=1.1.2", "numpy>=1.18.5", 
        "scipy>=1.5.2", "matplotlib>=3.3.2", "seaborn>=0.11.0",
        "xlrd>=1.2.0", "tensorflow-hub>=0.9.0", 
        "tensorflow-addons>=0.11.2",
        "tensorflow>=2.4.0", "transformers>=3.2.0, <=3.5.1",
        "seqeval>=0.0.17", "flask>=1.1.2"
    ],
    extras_require=extras,
    python_requires=">=3.6.0",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence, NLP, NLU",
    ],
)