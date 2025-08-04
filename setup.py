from setuptools import setup, find_packages

setup(
    name='mobile_language_mapping',
    version='0.1.0',
    packages=find_packages(),
    author='Your Name',
    author_email='your.email@example.com',
    description='A short description of the project.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='http://your/project/url',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)