# Contributing to Play with NLP

Thanks for your interest in adding a new NLP experiment! This guide will help you set up your project correctly.

## Experiment Structure

Each experiment should follow this structure:

```
experiments/
└── your-experiment-name/
    ├── README.md           # Experiment documentation
    ├── experiment.py       # Main implementation code
    ├── notebook.ipynb      # Interactive Jupyter notebook
    ├── requirements.txt    # (Optional) Specific dependencies
    └── data/              # (Optional) Sample data or data loading scripts
```

## Creating a New Experiment

### 1. Create Your Experiment Folder

```bash
mkdir -p experiments/your-experiment-name
cd experiments/your-experiment-name
```

### 2. Add a README.md

Your README should include:

```markdown
# Experiment Name

## Overview
Brief description of what this experiment does.

## Approach
- Key techniques or algorithms used
- Libraries and models utilized
- Any preprocessing steps

## Results
- Key findings
- Performance metrics (if applicable)
- Sample outputs or visualizations

## Usage

### Running the Code
\```bash
python experiment.py
\```

### Running the Notebook
\```bash
jupyter notebook notebook.ipynb
\```

## References
- Link to papers, tutorials, or resources used
```

### 3. Implement Your Code

Create `experiment.py` with:
- Clear function definitions
- Comments explaining key steps
- Example usage in `if __name__ == "__main__":` block

### 4. Create an Interactive Notebook

Your `notebook.ipynb` should:
- Walk through the experiment step-by-step
- Include visualizations
- Show example outputs
- Explain key concepts

### 5. Update Main README

Add your experiment to the main repository README.md under the "Projects" section.

## Code Quality Guidelines

- Write clean, readable code
- Add comments for complex logic
- Include docstrings for functions
- Use meaningful variable names
- Keep experiments self-contained

## Example Experiments

Check out existing experiments in the `experiments/` folder for reference:
- `sentiment-analysis/`
- `text-classification/`
- `named-entity-recognition/`

## Questions?

Open an issue if you need help or have questions about contributing!
