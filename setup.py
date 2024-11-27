import setuptools

requirements = []
with open('requirements.txt', 'rt') as f:
    for req in f.read().splitlines():
        if req.startswith('git+'):
            pkg_name = req.split('/')[-1].replace('.git', '')
            if "#egg=" in pkg_name:
                pkg_name = pkg_name.split("#egg=")[1]
            requirements.append(f'{pkg_name} @ {req}')
        else:
            requirements.append(req)

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyterrier-t5",
    version="0.0.1",
    author="Sean MacAvaney",
    author_email='sean.macavaney{at}.glasgow.ac.uk',
    description="PyTerrier components for T5 ranking",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    python_requires='>=3.9',
)
