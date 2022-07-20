from setuptools import setup
from setuptools import find_packages


def load(path):
	return open(path, 'r').read()


ml_ops_pipeline_version = '2.12.3'


classifiers = [
	"Development Status :: 5 - Production/Stable",
	"Environment :: Console",
	"Intended Audience :: Science/Research",
	"License :: OSI Approved :: MIT License",
	"Operating System :: OS Independent",
	"Programming Language :: Python",
	"Programming Language :: Python :: 3",
	"Topic :: Scientific/Engineering"]


if __name__ == "__main__":
	setup(
		name="ML Ops Pipeline",
		version=ml_ops_pipeline_version,
		description="ML Ops Pipeline",
		long_description=load('README.md'),
		long_description_content_type='text/markdown',
		platforms="OS Independent",
		package_data={'numerai': ['README.md']},
		packages=find_packages(exclude=['tests']),
		install_requires=["requests", "pytz", "python-dateutil",
						  "tqdm>=4.29.1", "click>=7.0", "pandas>=1.1.0"],
		entry_points={
		  'console_scripts': [
			  'numerapi = numerapi.cli:cli'
		  ]
		  },
		)