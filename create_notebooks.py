#!/usr/bin/env python
"""Script to create working Jupyter notebooks."""
import json
from pathlib import Path

def create_data_exploration_notebook():
    """Create the data exploration notebook."""
    notebook = {
        'cells': [
            {
                'cell_type': 'markdown',
                'metadata': {},
                'source': [
                    '# Data Exploration for Comp Recommendation System\n',
                    '\n',
                    'This notebook explores the appraisals dataset to understand the structure and prepare for model development.'
                ]
            },
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [
                    'import sys\n',
                    'import pandas as pd\n',
                    'import numpy as np\n',
                    'import matplotlib.pyplot as plt\n',
                    'import seaborn as sns\n',
                    'from pathlib import Path\n',
                    '\n',
                    '# Add src to path\n',
                    'sys.path.append(str(Path.cwd().parent / "src"))\n',
                    '\n',
                    'from utils.data_utils import load_appraisals_data\n',
                    '\n',
                    '%matplotlib inline\n',
                    'sns.set_style("whitegrid")'
                ]
            },
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [
                    '# Load the data\n',
                    'data_path = Path.cwd().parent / "data" / "appraisals_dataset.json"\n',
                    'df = load_appraisals_data(str(data_path))\n',
                    'print(f"Dataset shape: {df.shape}")\n',
                    'df.head()'
                ]
            },
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [
                    '# Display column names\n',
                    'print("Columns in dataset:")\n',
                    'for col in df.columns:\n',
                    '    print(f"  - {col}")'
                ]
            },
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [
                    '# Check data types\n',
                    'df.dtypes'
                ]
            },
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [
                    '# Basic statistics for numeric columns\n',
                    'df.describe()'
                ]
            },
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [
                    '# Examine subject property fields\n',
                    'subject_cols = [col for col in df.columns if col.startswith("subject.")]\n',
                    'print(f"\\nSubject property fields ({len(subject_cols)}):")\n',
                    'for col in subject_cols[:10]:  # First 10\n',
                    '    print(f"  - {col}")'
                ]
            },
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [
                    '# Examine a sample subject property\n',
                    'print("Sample subject property (first appraisal):")\n',
                    'for col in subject_cols[:5]:\n',
                    '    print(f"{col}: {df[col].iloc[0]}")'
                ]
            },
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [
                    '# Examine properties structure\n',
                    'print("Properties field (first appraisal):")\n',
                    'print(f"Type: {type(df[\\"properties\\"].iloc[0])}")\n',
                    'if isinstance(df["properties"].iloc[0], list):\n',
                    '    print(f"Number of properties: {len(df[\\"properties\\"].iloc[0])}")\n',
                    '    if len(df["properties"].iloc[0]) > 0:\n',
                    '        print(f"\\nSample property (first one):")\n',
                    '        sample = df["properties"].iloc[0][0]\n',
                    '        if isinstance(sample, dict):\n',
                    '            for key, value in list(sample.items())[:5]:\n',
                    '                print(f"  {key}: {value}")'
                ]
            },
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [
                    '# Examine comps structure\n',
                    'print("Comps field (first appraisal):")\n',
                    'print(f"Type: {type(df[\\"comps\\"].iloc[0])}")\n',
                    'if isinstance(df["comps"].iloc[0], list):\n',
                    '    print(f"Number of comps selected: {len(df[\\"comps\\"].iloc[0])}")\n',
                    '    if len(df["comps"].iloc[0]) > 0:\n',
                    '        print(f"\\nSample comp (first one):")\n',
                    '        sample = df["comps"].iloc[0][0]\n',
                    '        if isinstance(sample, dict):\n',
                    '            for key, value in list(sample.items())[:5]:\n',
                    '                print(f"  {key}: {value}")'
                ]
            },
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [
                    '# Distribution of number of properties per appraisal\n',
                    'num_properties = df["properties"].apply(lambda x: len(x) if isinstance(x, list) else 0)\n',
                    'plt.figure(figsize=(10, 6))\n',
                    'plt.hist(num_properties, bins=20, edgecolor="black", alpha=0.7)\n',
                    'plt.xlabel("Number of Properties")\n',
                    'plt.ylabel("Frequency")\n',
                    'plt.title("Distribution of Number of Properties per Appraisal")\n',
                    'plt.grid(axis="y", alpha=0.3)\n',
                    'plt.show()\n',
                    '\n',
                    'print(f"Average number of properties: {num_properties.mean():.2f}")\n',
                    'print(f"Min: {num_properties.min()}, Max: {num_properties.max()}")'
                ]
            },
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [
                    '# Distribution of number of comps selected per appraisal\n',
                    'num_comps = df["comps"].apply(lambda x: len(x) if isinstance(x, list) else 0)\n',
                    'plt.figure(figsize=(10, 6))\n',
                    'plt.hist(num_comps, bins=20, edgecolor="black", alpha=0.7, color="orange")\n',
                    'plt.xlabel("Number of Comps Selected")\n',
                    'plt.ylabel("Frequency")\n',
                    'plt.title("Distribution of Number of Comps Selected per Appraisal")\n',
                    'plt.grid(axis="y", alpha=0.3)\n',
                    'plt.show()\n',
                    '\n',
                    'print(f"Average number of comps selected: {num_comps.mean():.2f}")\n',
                    'print(f"Min: {num_comps.min()}, Max: {num_comps.max()}")'
                ]
            },
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [
                    '# Selection rate: comps / properties\n',
                    'selection_rate = num_comps / num_properties\n',
                    'plt.figure(figsize=(10, 6))\n',
                    'plt.hist(selection_rate, bins=20, edgecolor="black", alpha=0.7, color="green")\n',
                    'plt.xlabel("Selection Rate (Comps / Properties)")\n',
                    'plt.ylabel("Frequency")\n',
                    'plt.title("Distribution of Comp Selection Rate")\n',
                    'plt.grid(axis="y", alpha=0.3)\n',
                    'plt.show()\n',
                    '\n',
                    'print(f"Average selection rate: {selection_rate.mean():.2%}")'
                ]
            }
        ],
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            },
            'language_info': {
                'codemirror_mode': {
                    'name': 'ipython',
                    'version': 3
                },
                'file_extension': '.py',
                'mimetype': 'text/x-python',
                'name': 'python',
                'nbconvert_exporter': 'python',
                'pygments_lexer': 'ipython3',
                'version': '3.9.0'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 5
    }

    return notebook


def create_quick_view_notebook():
    """Create a quick view notebook."""
    notebook = {
        'cells': [
            {
                'cell_type': 'markdown',
                'metadata': {},
                'source': [
                    '# Quick View - Comp Recommendation Data\n',
                    '\n',
                    'A quick look at the dataset structure.'
                ]
            },
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [
                    'import sys\n',
                    'import pandas as pd\n',
                    'from pathlib import Path\n',
                    '\n',
                    'sys.path.append(str(Path.cwd().parent / "src"))\n',
                    'from utils.data_utils import load_appraisals_data'
                ]
            },
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [
                    'data_path = Path.cwd().parent / "data" / "appraisals_dataset.json"\n',
                    'df = load_appraisals_data(str(data_path))\n',
                    'print(f"Shape: {df.shape}")\n',
                    'df.head(2)'
                ]
            },
            {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': [
                    'df.columns.tolist()'
                ]
            }
        ],
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            },
            'language_info': {
                'codemirror_mode': {
                    'name': 'ipython',
                    'version': 3
                },
                'file_extension': '.py',
                'mimetype': 'text/x-python',
                'name': 'python',
                'nbconvert_exporter': 'python',
                'pygments_lexer': 'ipython3',
                'version': '3.9.0'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 5
    }

    return notebook


def main():
    """Create both notebooks."""
    notebooks_dir = Path('notebooks')
    notebooks_dir.mkdir(exist_ok=True)

    # Create data exploration notebook
    exploration_nb = create_data_exploration_notebook()
    exploration_path = notebooks_dir / 'data_exploration.ipynb'
    with open(exploration_path, 'w', encoding='utf-8') as f:
        json.dump(exploration_nb, f, indent=1, ensure_ascii=False)
    print(f'Created {exploration_path}')

    # Create quick view notebook
    quick_nb = create_quick_view_notebook()
    quick_path = notebooks_dir / 'quick_view.ipynb'
    with open(quick_path, 'w', encoding='utf-8') as f:
        json.dump(quick_nb, f, indent=1, ensure_ascii=False)
    print(f'Created {quick_path}')

    print('\nNotebooks created successfully!')


if __name__ == '__main__':
    main()
