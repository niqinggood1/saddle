import setuptools
#from setuptools import setup, find_packages
from distutils.core import setup
#setuptools.
setup(
    name="saddle",
    version="1.0.0",
    author="patrick",
    author_email="niqinggood@gmail.com",
    description="Use this package to make Machine Learn Algorithm more convenient ",
    long_description="made by Patrick",
    long_description_content_type="text/markdown",
    url='https://github.com/niqinggood1/saddle',
    #packages=['ctr','dl','nlp','feature_process','statistical_model','timeseries','utility'],
    packages= setuptools.find_packages( exclude=["build*", "test*", "examples*"] ), #["saddle"], # #,"dist*","saddle.egg-info"
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data = True,
    python_requires='>=3.6',
    install_requires=[
        
        #'tqdm ~= 4.49.0',
        #'numpy >= 1.15.0'
    ], 
)