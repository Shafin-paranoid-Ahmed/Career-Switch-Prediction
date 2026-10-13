# Career Switch Prediction

Predicts whether a professional will switch careers from demographic, education and work-history features. This was a course project for CSE422 (Artificial Intelligence) at BRAC University.

## Data

`Career_Switch_Prediction_Dataset.csv` has 5,000 rows, 12 features and a binary target `will_change_career`.
- **Imbalance:** 25.2% of people switch careers, so a model that always predicts "won't switch" already reaches 74.7% accuracy. Every model below is compared against that majority-class baseline, using macro-F1 and ROC-AUC in addition to accuracy.
- **Source:** the course provided the data. Its columns match the public Kaggle dataset "HR Analytics: Job Change of Data Scientists".

## Method

- **Cleaning:** the script drops `enrollee_id` and converts `experience` to numbers (`<1` → 0.5, `>20` → 21). It also fixes the `Oct-49` spreadsheet artefact back to `10-49` in `company_size`.
- **Split:** 70/30 train/test, stratified, with seed 42.
- **Preprocessing:** the median imputer, the scaler and the one-hot encoder are fitted on the **training split only**, inside each model's pipeline. Missing categorical values get their own `missing` category.
- **Models:**
  - logistic regression
  - KNN (k = 5)
  - decision tree
  - neural network: 128 → 64 ReLU layers, sigmoid output, binary cross-entropy, Adam, early stopping on a 20% validation split

## Results

Test set, 1,500 rows. Precision and recall are for the "will switch" class.

| Model | Accuracy | Precision | Recall | Macro-F1 | ROC-AUC |
|---|---|---|---|---|---|
| Majority-class baseline | 0.747 | 0.000 | 0.000 | 0.428 | 0.500 |
| Logistic regression | 0.783 | 0.596 | 0.443 | 0.685 | 0.767 |
| KNN (k = 5) | 0.758 | 0.524 | 0.467 | 0.667 | 0.727 |
| Decision tree | 0.721 | 0.451 | 0.470 | 0.636 | 0.638 |
| Neural network | 0.792 | 0.627 | 0.435 | 0.691 | 0.777 |

**Takeaways:**
- All four models beat the baseline on macro-F1.
- Logistic regression and the neural network are the strongest. Their small lead does not justify the network's extra complexity.
- Every model catches fewer than half of the people who do switch careers (recall ≤ 0.47).
- Class weighting or threshold tuning are the obvious next steps.

## Run it

Python 3.12:

```bash
pip install -r requirements.txt
python Career_Switch_Classification.py           # prints the results table
python Career_Switch_Classification.py --plots   # also shows the exploratory plots
```

## Report

[`Career_Switch_Classification_Report.pdf`](Career_Switch_Classification_Report.pdf) is the report as originally submitted. Its numbers come from the original notebook, so they differ slightly from the table above.

## Authors

Group 17: Anindya Dasgupta and Shafin Ahmed.
