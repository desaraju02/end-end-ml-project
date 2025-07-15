from typing import List
from setuptools import setup, find_packages

HYPHEN_E_DOT = '-e .'
def get_requirements(requirements_file:str) -> List[str]:
    """
    This function reads the requirements file and returns a list of requirements.
    """
    requirements = []
    with open(requirements_file, 'r') as file:
        requirements = file.readlines()
        requirements = [req.replace('\n','') for req in requirements]   
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements
setup(
    name='end-end-ml-project',
    packages=find_packages(),
    version='0.1',
    author='Nages Desaraju',
    install_requires=get_requirements('requirements.txt'),
)