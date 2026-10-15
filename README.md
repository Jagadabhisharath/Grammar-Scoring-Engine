# Grammar Scoring Engine from Spoken English Audio

Predicts a continuous **grammar score (0 to 5)** from spoken-English WAV recordings using acoustic feature extraction and scikit-learn regression.

It is a **baseline** pipeline on a small labelled dataset (769 recordings). Models are compared against a mean-prediction baseline with 5-fold cross-validation, the strongest are tuned with `GridSearchCV`, and both cross-validated and in-sample error are reported so results are not overstated.

## Results

| Model | CV RMSE (5-fold, mean ± std) |
| --- | --- |
| **Tuned SVR (RBF), selected** | **0.703 ± 0.028** |
| SVR (RBF) baseline | 0.727 ± 0.036 |
| Extra Trees | 0.752 ± 0.045 |
| Random Forest | 0.787 ± 0.042 |
| Hist Gradient Boosting | 0.794 ± 0.034 |
| Tuned Ridge | 0.845 ± 0.059 |
| Ridge baseline | 0.850 ± 0.064 |
| Dummy mean baseline | 1.240 ± 0.063 |

- Tuned SVR (best parameters: `C=10`, `epsilon=0.05`, `gamma=0.05`) cuts cross-validated RMSE by about **43%** versus predicting the mean (1.24 to 0.70).
- **Training RMSE: 0.0481** (MAE 0.0472, R² 0.9985). This is measured on the data the final model was fitted on, so it is an in-sample figure and **not** a generalization estimate. The gap between 0.05 (training) and 0.70 (cross-validation) shows the final model overfits the small training set.

## Pipeline

1. **Load and inspect** `train.csv`, `test.csv` and `sample_submission.csv`.
2. **Extract audio** from `train.zip` and `test.zip` if present.
3. **Data-quality checks:** assert required columns, no missing or non-numeric labels, no duplicate filenames; inspect the label distribution; verify every listed WAV file exists.
4. **Feature extraction** (librosa, mono, 16 kHz, leading/trailing silence trimmed): 53 numeric features per file.
   - duration and RMS energy
   - zero-crossing rate (mean, std)
   - 20 MFCCs (mean, std)
   - spectral centroid, bandwidth and rolloff
   - a rough tempo estimate
5. **Optional transcript features** (`faster-whisper` plus simple text-structure features). Disabled in the reported run, so all results above use acoustic features only.
6. **Model comparison and tuning:** median imputation, scaling for Ridge and SVR, six models plus a mean baseline with shuffled 5-fold cross-validation (`random_state=42`), then `GridSearchCV` on SVR and Ridge (400 fits for SVR).
7. **Final fit and report:** refit the selected model on all training data; report training RMSE, MAE and R² with actual-vs-predicted and residual plots.
8. **Prediction validation:** predictions are clipped to 0 to 5 and checked for count, order, uniqueness, missing values and range before saving `submission_corrected.csv`.
9. **Feature importance** from a separate Extra Trees model. Importances show which acoustic features the model used, not that they measure grammar.

## Data notes

- 769 labelled training rows and 216 rows in `test.csv`.
- Labels range from 0 to 5 (observed values: 0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5) and are used exactly as supplied.
- **Submission mismatch:** `test.csv` lists 216 filenames but `sample_submission.csv` lists 204, and only 25 filenames overlap. The notebook reports this and prints a warning not to upload predictions until the expected format is confirmed.
- The dataset is **not included** in this repository.

## Setup and usage

1. Install dependencies:

   ```bash
   pip install librosa soundfile scikit-learn pandas numpy matplotlib
   pip install faster-whisper   # optional, only for transcript features
   ```

2. Place the data in a folder named `Dataset_Final` next to the notebook (or edit the candidate paths in the configuration cell):

   ```
   Dataset_Final/
     train.csv
     test.csv
     sample_submission.csv
     train.zip    # extracted to Dataset_Final/train
     test.zip     # extracted to Dataset_Final/test
   ```

3. Open `grammar_scoring_engine_tuned_and_validated.ipynb` and run all cells in order.

## Limitations and known issues

- The features describe **audio signal and vocal delivery** (timbre, energy, pacing), not words or grammar directly, so the model likely relies on correlated delivery cues.
- With 769 samples, scores vary by fold. The very low training RMSE against the CV RMSE indicates overfitting.
- There are no test-set labels, so there is no true held-out score.
- The notebook warns about **duplicate WAV basenames** across the train and test folders (773 unique names found). Audio lookup should be restricted to the train folder for training rows and the test folder for test rows; verify this before trusting test predictions.
- The filename mismatch between `test.csv` and `sample_submission.csv` is unresolved.

## Ideas for improvement

- Enable the transcript features and compare their CV RMSE against the acoustic-only baseline.
- Try embeddings from a pretrained speech model in place of hand-crafted MFCC statistics.
- Use nested cross-validation for tuning and stronger regularization to reduce the overfitting gap.
- Add MAE, correlation with human scores and per-score-band error analysis.

## Tech stack

Python, librosa, scikit-learn, pandas, NumPy, Matplotlib, faster-whisper (optional)
