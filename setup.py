from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name='fake-news-detector',
    packages=find_packages(),
    version='1.0.0',
    description='An AI-powered fake news detection system using hybrid BERT models and advanced NLP techniques.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Ashkan Nadafian',
    license='MIT',
    python_requires='>=3.8',
    install_requires=requirements,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing :: Linguistic',
    ],
    keywords='fake-news detection nlp bert transformers pytorch machine-learning',
    project_urls={
        'Source': 'https://github.com/yourusername/fake-news-detector',
        'Documentation': 'https://github.com/yourusername/fake-news-detector#readme',
    },
)
