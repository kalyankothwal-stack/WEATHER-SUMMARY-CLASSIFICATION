# Weather Summary Classification

Predict the daily weather summary (Partly Cloudy, Foggy, Rainy and so on) from hourly weather readings, and serve the model in a Streamlit app.

**Live demo:** [weather-summary-classification.streamlit.app](https://weather-summary-classification.streamlit.app/)

## The problem

The dataset has about 96,400 hourly weather readings (2006 to 2016, 4,018 days). Each day carries a text summary with 214 different labels. The task is to predict that summary from temperature, humidity, wind, visibility, pressure, precipitation type and month.

## What I did

- **Grouped 214 labels into 7 categories** with keyword rules (Clear, Partly Cloudy, Mostly Cloudy, Overcast, Foggy, Rainy, Breezy), and checked the groups against the weather measurements.
- **Split by date, not by row.** Every hour of a day shares one label, so a random row split puts hours of the same day in both train and test and inflates the score. I split days 70 / 15 / 15 into train, validation and test.
- **Fitted all preprocessing on the train set only:** filling missing values, outlier capping (IQR bounds), one-hot encoding and scaling. Pressure and humidity readings of exactly 0 are not real, so I treated them as missing.
- **Removed data leakage:** dropped the `Summary` column, which describes the weather directly.
- **Tested the features instead of trusting the importance plot.** Date features (year, day, day of year, week, hour) ranked high but only helped on random days, not when predicting later years, because they let the model recognise neighbouring days. I removed them, along with interaction and temporal features that added nothing. The final model uses 9 inputs.
- **Compared 11 models** with 5-fold `StratifiedGroupKFold` on the train set (days stay together, every category appears in every fold), scored by macro F1 with class weights for the rare categories.
- **Tuned the best model** with randomized search on the same folds.
- **Evaluated on the test set once**, at the end.

### Features used

Temperature, apparent temperature, humidity, wind speed, wind direction, visibility, pressure, month and precipitation type (snow or not).

### Model comparison (5-fold cross-validated macro F1 on the train set)

| Model | CV macro F1 |
|---|---|
| Hist Gradient Boosting | 0.317 |
| XGBoost | 0.315 |
| Gradient Boosting | 0.314 |
| LightGBM | 0.308 |
| Random Forest | 0.304 |
| K-Nearest Neighbors | 0.293 |
| Decision Tree | 0.274 |
| Naive Bayes | 0.262 |
| AdaBoost | 0.237 |
| Logistic Regression | 0.229 |
| SVM (linear) | 0.197 |

The top five are within the fold-to-fold variation (about 0.02), so they are effectively tied. I took Hist Gradient Boosting forward because it had the best score and trains fast. Tuning moved its cross-validated macro F1 from 0.317 to 0.324, which is within the noise.

## Results on the test set

| | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Tuned Hist Gradient Boosting | 0.545 | 0.302 | 0.580 |
| Baseline (always predict Partly Cloudy) | 0.443 | 0.088 | 0.272 |

| Category | F1 | AUC |
|---|---|---|
| Foggy | 0.68 | 0.89 |
| Partly Cloudy | 0.67 | 0.85 |
| Mostly Cloudy | 0.46 | 0.71 |
| Overcast | 0.17 | 0.78 |
| Clear | 0.08 | 0.84 |
| Breezy | 0.05 | 0.76 |
| Rainy | 0.00 | 0.48 |

Macro F1 is the main score because the categories are very uneven. Always predicting the most common one gives 44% accuracy but a macro F1 of only 0.09.

## Limitations

- The label describes the whole day's sky, but each row is one hour of readings and the data has no cloud-cover measurement. This limits every model, especially on Mostly Cloudy against Partly Cloudy.
- Rainy, Breezy and Clear have very few days (Rainy: 7 in train, 2 in test), so their scores are unreliable. The model's probabilities for them are inflated by the class weights.
- An earlier version of this project used a random row split and date features, which gave higher scores. Those scores did not hold on unseen days, so the numbers here are lower and honest.

## The app

`weather_app.py` is a Streamlit app. Enter one hour of readings and it shows the predicted category, the model probability and the probability for every category. Inputs outside the range seen in training are clipped to that range, as in training.

## Run it locally

```
pip install -r requirements.txt
streamlit run weather_app.py
```

`weather_model.pkl` must be in the same folder. It is created by the last cells of the notebook and holds the model, the scaler, the feature order, the capping bounds and the category names.

To run the notebook itself, also install:

```
pip install xgboost lightgbm matplotlib seaborn plotly jupyter
```

## Project files

- `weather_app.py`: the Streamlit app
- `weather_model.pkl`: the saved model and preprocessing
- `requirements.txt`: packages for the app
- `WEATHER-SUMMARY-CLASSIFICATION-PROJECT-BY-KALYAN.ipynb`: the full analysis, from data loading to the final test evaluation

## Tools

Python, pandas, NumPy, scikit-learn, XGBoost, LightGBM, Plotly, Streamlit.

## Author

Kalyan