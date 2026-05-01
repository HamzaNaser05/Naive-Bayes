# Naive Bayes Student Grade Predictor

## Description
This project implements a Naive Bayes classifier to predict student grades based on various features like gender, transportation, accommodation, preparation for midterms, and note-taking in classes. It uses Laplace smoothing to handle zero probabilities and evaluates the model on holdout and test datasets.

## Files
- `Naive_Assignment.py`: The main Python script containing the implementation.
- `Naive_Assignment.ipynb`: A Jupyter notebook version of the script for interactive execution.
- `csv_students_naive_train.csv`: Training dataset with student features and grades.
- `csv_students_naive.csv`: Test dataset (if used separately).

## Requirements
- Python 3.x
- pandas
- matplotlib

Install dependencies with: `pip install pandas matplotlib`

## How to Run
1. Ensure the CSV files are in the same directory as the script/notebook.
2. Run the Python script: `python Naive_Assignment.py`
   Or open and run the Jupyter notebook: `Naive_Assignment.ipynb`

## Output
The script will:
- Split the data into training, holdout, and test sets.
- Train the Naive Bayes model with different Laplace smoothing values (k=0,1,2,3).
- Plot the accuracy on the holdout set.
- Select the best model and evaluate it on the test set.
- Print the results, including the best k value and final test accuracy.
