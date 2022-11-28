from distutils.core import setup

setup(
    name = 'rfsd',
    version='0.1',
    description="Linear-time random feature Stein discrepancies",
    author='Jonathan H. Huggins',
    author_email='jhuggins@mit.edu',
    url='https://bitbucket.org/jhhuggins/random-feature-stein-discrepancies/',
    packages=['rfsd'],
        install_requires=[
            'numpy', 'scipy', 'matplotlib', 'autograd',
            'sklearn', 'seaborn', 'future'],
    keywords = ['goodness-of-fit testing', "Stein's method",
                'kernel Stein discrepancies'],
    platforms='ALL',
)
