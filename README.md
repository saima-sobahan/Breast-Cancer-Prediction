# Breast Cancer Prediction

A machine learning project that classifies breast tumors as **benign** or **malignant** from cell nucleus measurements. Built as the lab project for **CSE422: Artificial Intelligence** at BRAC University (Section 11, Group 21).

## Dataset

- **Source:** [Breast Cancer Wisconsin (Diagnostic) dataset on Kaggle](https://www.kaggle.com/code/lbronchal/breast-cancer-dataset-analysis/input)
- **Size:** 569 samples, 33 columns
- **Target:** `diagnosis`. The data has 357 benign samples and 212 malignant samples, so the classes are imbalanced.
- **Features:** the mean, standard error and "worst" value of 10 tumor measurements: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension.

## Method

1. **Cleaning:** dropped the empty `Unnamed: 32` column, which is null in all 569 rows.
2. **Encoding:** changed `diagnosis` into numbers, with Malignant as `1` and Benign as `0`.
3. **Feature scaling:** applied `RobustScaler` to the features because it handles outliers well. The `id` column was not scaled.
4. **Splitting:** used a 70% training / 30% test split. The split is stratified on the target (`stratify=y`, `random_state=42`) so both sets keep the same class balance.
5. **Models:** trained four classifiers:
   - Logistic Regression
   - K-Nearest Neighbors (`n_neighbors=3`)
   - Random Forest (`n_estimators=100`)
   - Gaussian Naive Bayes

## Results

| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|---------:|----------:|-------:|---------:|
| Logistic Regression | **98.83%** | 0.98    | 98.41% | 98.41%   |
| Random Forest       | 97.66%   | 0.98      | 95.24% | 96.77%   |
| KNN                 | 93.57%   | 0.93      | 88.88% | 91.05%   |
| Naive Bayes         | 93.57%   | 0.91      | 90.47% | 91.20%   |

### Confusion matrices (test set, 171 samples)

The rows are the actual class and the columns are the predicted class, in the order `[Benign, Malignant]`.

| Model               | Matrix                    |
|---------------------|---------------------------|
| Logistic Regression | `[[107, 1], [1, 62]]`     |
| Random Forest       | `[[107, 1], [3, 60]]`     |
| KNN                 | `[[104, 4], [7, 56]]`     |
| Naive Bayes         | `[[103, 5], [6, 57]]`     |

Logistic Regression performed best overall. It made only one false negative, meaning only one malignant tumor was predicted as benign. That matters most in cancer screening, because a missed malignant tumor is the most costly kind of error.

## How to Run

1. Install the required libraries:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```
2. Download the dataset (`data.csv`) from the Kaggle link above.
3. Open the notebook and update the file path in the `read_csv` cell:
   ```bash
   jupyter notebook CSE422_Project.ipynb
   ```
4. Run all cells.

## Files

- `CSE422_Project.ipynb`: the full notebook, covering data analysis, preprocessing, model training and evaluation.

## Authors

- Saima Sobahan
- Md. Ashiqur Rahman Abir

## Note

This is an academic project. It is not meant for real medical diagnosis.
