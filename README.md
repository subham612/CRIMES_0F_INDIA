# Crimes of India: Data Analysis (2014–2016)

## Overview

A Python data-analysis project exploring state-wise and union-territory-wise
Indian Penal Code (IPC) crime statistics for 2014, 2015, and 2016.

The project uses data cleaning, validation, exploratory data analysis,
statistical summaries, and visualizations to examine crime counts and rates.

## Project Structure

- `main.py` — data processing, analysis, validation, and output generation
- `data/raw/` — original CSV dataset
- `data/processed/` — cleaned dataset copy
- `processed/` — processed dataset used by the analysis
- `notebooks/` — interactive Jupyter Notebook report
- `output/` — generated charts, CSV summaries, and findings
- `.gitignore` — files excluded from Git tracking

## Requirements

- Python 3
- pandas
- NumPy
- Matplotlib
- Jupyter Notebook (for the interactive report)

## Run the Analysis

From the project directory, activate your virtual environment if needed,
then run:

```bash
python main.py
Generated reports and charts are saved in the `output/` directory.

To explore the notebook, open `notebooks/crimes_of_india_analysis.ipynb`
in Jupyter Notebook or VS Code.

## Scope and Limitations

The analysis covers IPC crime statistics for 2014–2016 only.
Reported crime counts and crime rates measure different things.
Recorded crime statistics do not represent every crime that occurred,
and associations in the data do not establish causation.

## Data Source

The raw dataset is named `Table_1A.1_2016.csv` and contains
state/UT-wise IPC crime statistics for 2014–2016.
Consult the original publication and its terms for source attribution
and data reuse information.

## Outputs

The `output/` directory contains visualizations, summary tables,
crime-rate rankings, year-to-year comparisons, and written findings.
