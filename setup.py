from setuptools import setup, find_packages

setup(
    name='DFTDescriptorPipeline',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'openpyxl',
        'morfeus-ml',
        'joblib'
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A pipeline to extract quantum chemical descriptors from Gaussian log files and perform regression analysis.',
    url='https://github.com/yourusername/DFTDescriptorPipeline',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
