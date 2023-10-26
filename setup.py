from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='topicphrase',
      version='0.2.0',
      description='Keyphrases extraction and topic modeling with Sentence Transformers',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Sergei Rybalko',
      author_email='benja1972@gmail.com',
      license='BSD',
      packages=find_packages(),
      url="https://github.com/Benja1972/topicphrase",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
      install_requires=[
        'hdbscan>=0.8.33',
        'numpy>=1.25.0',
        'pke>=2.0.0',
        'sentence_transformers>=2.2.2',
        'spacy>=3.2.3',
        'umap-learn>=0.5.4',
      ],
       python_requires='>=3.8',
      )
