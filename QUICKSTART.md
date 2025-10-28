# Quick Start Guide

Get up and running with the Comp Recommendation System in 3 simple steps!

## Prerequisites

- Python 3.9 or higher
- pip package manager

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Clean the Data

```bash
python run.py clean
```

This will:
- Load the appraisal data from `data/appraisals_dataset.json`
- Process and clean the data
- Save the cleaned output to `data/cleaned_appraisals.csv`

### 3. Explore the Data

```bash
jupyter notebook notebooks/data_exploration.ipynb
```

Run all cells to see:
- Dataset statistics
- Data structure analysis
- Distribution of properties and comps
- Visualizations

## What's Working

The project is now fully runnable with:

- **Data Loading**: Handles nested JSON structure with 88 appraisals
- **Data Cleaning**: Standardizes columns, checks duplicates, reports missing values
- **CLI Interface**: Simple command-line interface via `run.py`
- **Exploration Notebook**: Interactive analysis in Jupyter
- **Proper Package Structure**: All modules properly organized with `__init__.py` files

## Next Steps

Now that the foundation is working, you can:

1. **Analyze the data** - Run the exploration notebook to understand property features
2. **Feature Engineering** - Extract and create relevant features for modeling
3. **Build Models** - Implement Milestone 1: Statistical Modeling
   - Clustering algorithms
   - Nearest Neighbors
   - Other statistical methods
4. **Add Explainability** - Implement Milestone 2: LLM-based explanations
5. **Create Feedback Loop** - Implement Milestone 3: Self-improving system

## Troubleshooting

### Import Errors

If you get import errors, make sure you're in the project root directory:
```bash
cd c:\Users\rodri\comp-recommendation-system
```

### Missing Dependencies

Install all requirements:
```bash
pip install -r requirements.txt
```

### Data Not Found

Ensure `data/appraisals_dataset.json` exists in the project directory.

## Available Commands

```bash
# Clean data
python run.py clean

# View help
python run.py --help

# Future commands (not yet implemented)
python run.py train
python run.py predict
python run.py evaluate
```

## Project Status

**Current Version**: 0.1.0

**Completed**:
- ✅ Data loading pipeline
- ✅ Data cleaning and preprocessing
- ✅ Basic exploration tools
- ✅ CLI interface

**In Development**:
- 🔄 Feature engineering
- 🔄 Model training

**Planned**:
- 📋 Statistical modeling (Milestone 1)
- 📋 Explainability features (Milestone 2)
- 📋 Self-improving system (Milestone 3)

---

**Ready to continue?** Check out the full [README.md](README.md) for more details!
