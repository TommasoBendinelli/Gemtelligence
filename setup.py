from setuptools import setup, find_packages

setup(
    name="csem_gemintelligence",
    version="0.0.1",
    author="CSEM",
    packages=find_packages("src/"),
    package_dir={"": "src/"},
    install_requires=[
        "wheel",
        "matplotlib",
        "scipy",
        "tsaug",
        "xlrd",
        "scikit-learn",
        "geopy",
        "pytest_notebook",
        "tqdm",
        "plotly",
        "ray",
        "click",
        "tqdm",
        "striprtf",
        "openpyxl",
        "more_itertools",
        "transformers",
        "seaborn",
        "hydra-core",
        "coloredlogs",
        "pytorch-lightning<=1.6",
        "einops",
        "pandas<1.5",
        "numpy<1.23",
        'streamlit==1.21.0',
        "protobuf~=3.19.0",
        "tables"
    ],
    # entry_points={
    #     "console_scripts": [
    #         "clean_folder=CSEM_ExperimentTracker.load_files:delete_empty_folder",
    #         "save_df = CSEM_ExperimentTracker.load_files:save_df",
    #     ]
    # },
)
