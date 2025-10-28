# Comp Recommendation System

A machine learning system for recommending comparable properties from appraisal datasets.

## Project Overview

This system analyzes appraisal data to recommend the best comparable properties (comps) for real estate appraisals. Each appraisal contains:
- Subject property details
- A list of available comparable properties
- The comps that were ultimately selected by an appraiser

The goal is to build a model that can learn from past selections to recommend optimal comps for new properties.

## Project Structure

```
.
├── data/                      # Data directory
│   ├── appraisals_dataset.json    # Raw appraisal data
│   └── cleaned_appraisals.csv     # Cleaned data output
├── src/                       # Source code
│   ├── data_cleaning/        # Data cleaning scripts
│   │   ├── __init__.py
│   │   └── clean_appraisals.py
│   ├── models/               # ML models (to be implemented)
│   │   └── __init__.py
│   └── utils/                # Utility functions
│       ├── __init__.py
│       └── data_utils.py
├── notebooks/                # Jupyter notebooks for analysis
│   └── data_exploration.ipynb
├── run.py                    # Main entry point
├── requirements.txt          # Project dependencies
└── README.md                # This file
```

## Setup

### 1. Clone the repository
```bash
cd comp-recommendation-system
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Main Entry Point

The project includes a main script ([run.py](run.py)) that provides a CLI interface:

```bash
# Clean and preprocess the data
python run.py clean

# Train models (not yet implemented)
python run.py train

# Make predictions (not yet implemented)
python run.py predict

# Evaluate model performance (not yet implemented)
python run.py evaluate

# Get help
python run.py --help
```

### Data Cleaning

The data cleaning process includes:
- Loading nested JSON data structure
- Standardizing property data and column names
- Handling duplicates between comps and properties
- Identifying missing values
- Data validation and quality checks

To run data cleaning directly:
```bash
python src/data_cleaning/clean_appraisals.py
```

Output will be saved to `data/cleaned_appraisals.csv`.

### Data Exploration

Use the Jupyter notebook for interactive data exploration:

```bash
jupyter notebook notebooks/data_exploration.ipynb
```

## Current Status

**Completed:**
- Project structure setup
- Data loading with nested JSON support
- Data cleaning pipeline
- Basic data exploration notebook
- Main CLI entry point

**In Progress:**
- Data analysis and feature engineering

**To Do (Milestones):**
1. **Statistical Modeling** - Implement scoring system using clustering, nearest neighbors, etc.
2. **Explainability** - Add LLM-based explanations for recommendations
3. **Self-Improving System** - Incorporate human feedback and continuous learning

## Dataset Information

The dataset contains **88 appraisals**, each with:
- **38 columns** including subject property features
- Variable number of available properties per appraisal
- Variable number of selected comps per appraisal

Key fields:
- `orderid` - Unique appraisal identifier
- `subject.*` - Subject property features (address, GLA, year built, etc.)
- `properties` - List of all available comparable properties
- `comps` - List of comps selected by the appraiser

## Development

### Running Tests
```bash
# Tests to be added
pytest
```

### Code Style
The project follows PEP 8 style guidelines.

## License

This project is for educational and research purposes.