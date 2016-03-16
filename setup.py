from setuptools import setup, find_packages

setup(
        name='pyspark-lsh',
        version='0.1',
        author='Shannon Quinn',
        description='Locality-sensitive hashing for Apache Spark. Largely a PySpark port of the spark-hash project.',
        url='https://github.com/wan/pyspark-lsh',#this is forked; original is at https://github.com/magsol/pyspark-lsh
        packages=find_packages(),
        scripts=['driver.py'],
        license='Apache Software License',
        install_requires=['numpy']#technically also pyspark, but that is not a pip dependency
)
