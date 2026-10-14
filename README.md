# Grammar Scoring Engine from Spoken English Audio

Predicts a continuous **grammar score (0 to 5)** from short spoken-English WAV recordings, using acoustic feature extraction and scikit-learn regression models.

This is a **baseline** pipeline built on a small labelled dataset (769 recordings). It compares several models against a mean-prediction baseline with 5-fold cross-validation and reports both cross-validated and in-sample error, so the results are not overstated.

## Results

| Model | CV RMSE (5-fold, mean ± std) |
| --- | --- |
| **Extra Trees (selected)** | **0.814 ± 0.045** |
| SVR (RBF) | 0.822 ± 0.063 |
| Random Forest | 0.840 ± 0.034 |
| Hist Gradient Boosting | 0.857 ± 0.025 |
| Ridge | 0.928 ± 0.040 |
| Dummy mean baseline | 1.240 ± 0.063 |

- The selected model cuts cross-validated RMSE by about **34%** versus predicting the mean (1.24 to 0.81).
- **Training RMSE: 0.1056** (MAE 0.0753, R² 0.9927). This is measured on the same data the final model was fitted on, so it is an in-sample figure and **not** a generalization estimate. The gap between 0.11 (training) and 0.81 (cross-validation) shows the final model overfits the small training set.

## Pipeline

1. **Load and inspect** `train.csv`, `test.csv` and `sample_submission.csv`.
2. **Extract and locate audio**: unzip `train.zip` and `test.zip`, find all WAV files, and confirm every listed file exists.
3. **Data-quality checks**: assert required columns, no missing labels, no duplicate training filenames; inspect the label distribution.
4. **Feature extraction** (librosa, mono, 16 kHz, leading/trailing silence trimmed): 53 numeric features per file.
   - duration and RMS energy
   - zero-crossing rate (mean, std)
   - 20 MFCCs (mean, std)
   - spectral centroid, bandwidth and rolloff
   - a rough tempo estimate
5. **Model comparison**: median imputation, scaling for Ridge and SVR, then 6 models evaluated with shuffled 5-fold cross-validation (`random_state=42`).
6. **Final fit and report**: refit the best model on all training data and report training RMSE, MAE and R² with diagnostic plots.
7. **Predictions**: predict the files listed in `test.csv`, clip scores to the 0 to 5 range, and save a CSV.

## Data notes

- 769 labelled training rows, 216 rows in `test.csv`, 773 WAV files found in total.
- Labels range from 0 to 5 in half-point steps (observed values: 0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5). Labels are used exactly as supplied.
- **File mismatch:** `test.csv` lists 216 filenames but `sample_submission.csv` lists 204, and only 25 filenames overlap. The notebook reports this mismatch instead of assuming the files are compatible. Check the official files before submitting predictions anywhere.
- The dataset is **not included** in this repository.

## Setup and usage

The notebook is written for Google Colab but runs anywhere with Python 3.

1. Install dependencies:

   ```bash
   pip install librosa soundfile scikit-learn pandas numpy matplotlib seaborn
   ```

2. Put your data files in `/content` (or change `PROJECT_DIR` in the notebook):

   ```
   train.csv
   test.csv
   sample_submission.csv
   train.zip      # extracted to train_data/
   test.zip       # extracted to test_data/
   ```

3. Open `grammar-scoring-engine.ipynb` and run all cells in order. The final cell saves `submission_from_test_csv.csv`.

## Limitations

- The features describe the **audio signal and vocal delivery** (timbre, energy, pacing). They do not model words or grammar directly, so the model is likely picking up correlated delivery cues rather than grammatical correctness.
- With only 769 samples, cross-validation scores vary by fold, and no hyperparameter search was run.
- There are no test-set labels, so there is no true held-out score.
- The training/sample-submission filename mismatch is unresolved.

## Ideas for improvement

- Transcribe the audio with a speech-recognition model and add text-based grammar features (error counts, sentence structure, fluency markers).
- Try embeddings from a pretrained speech model in place of hand-crafted MFCC statistics.
- Tune hyperparameters with nested cross-validation and add regularization to reduce the overfitting gap.
- Evaluate with additional metrics (MAE, correlation with human scores) and per-score-band error analysis.

## Tech stack

Python, librosa, scikit-learn, pandas, NumPy, Matplotlib, Seaborn

