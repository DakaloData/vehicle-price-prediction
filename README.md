   **Live app:** https://dakalo-car-price.streamlit.app/
# Used Car Price Prediction

An end-to-end machine learning project that predicts the selling price of a used car from its age, engine size, power, fuel efficiency, brand, fuel type and transmission, and serves the model through an interactive Streamlit app.

Built as the major project for a Postgraduate Diploma in Data Science.

## Results

Three models were tuned with cross-validation on the training set only, then evaluated once on a held-out test set of 3,048 vehicles.

| Model | R² (log price) | R² (Rupees) | RMSE (Rupees) | MAE (Rupees) |
| --- | --- | --- | --- | --- |
| Ridge Regression | 0.872 | 0.604 | 715,464 | 168,016 |
| **Random Forest (selected)** | 0.927 | **0.689** | **633,853** | 113,538 |
| Gradient Boosting | 0.929 | 0.678 | 644,857 | 113,449 |

Random Forest was selected because it had the best RMSE and R² on the Rupee scale, which is the scale that matters for real pricing decisions. Gradient Boosting was marginally better in log space and effectively tied on MAE.

Typical error for the selected model: about ₹113,500 (MAE) on unseen vehicles.

## Key findings

- **Engine power and vehicle age drive price.** Together, `max_power` and `vehicle_age` account for about 90% of the Random Forest's feature importance.
- **Depreciation is non-linear.** Log-transforming the skewed variables strengthened the correlation between vehicle age and price from roughly -0.24 to -0.49.
- **The linear model is clearly weaker.** Ridge trails both tree models on every metric, so the data has non-linear structure.
- **One extreme vehicle drives the largest error.** Both tree models' worst miss was the same ₹39.5 million car, a genuine record deliberately kept during cleaning.

## What the notebook covers

1. Data loading and quality checks (15,411 records, 14 columns)
2. Cleaning: removed 167 duplicates and 4 physically impossible records, leaving 15,240 rows. Statistical outliers were kept when they reflected real vehicles, such as 7-seaters and premium cars
3. Exploratory data analysis: distributions, skewness, log transforms, correlation heatmap
4. Feature selection with correlation analysis and `SelectKBest`, which agreed on the same 10 features
5. Train/test split before scaling, with the scaler fitted on the training data only to prevent data leakage
6. Model training and tuning with `GridSearchCV` (Ridge) and `RandomizedSearchCV` (Random Forest, Gradient Boosting)
7. Evaluation on the held-out test set in both log space and Rupees
8. Deployment as a Streamlit web app

## The app

`app.py` loads a saved bundle containing the trained model, the fitted scaler, and the list of preprocessing steps, so the app applies exactly the same pipeline as training.

Only 3 brands (BMW, Maruti, Mercedes-Benz) and 2 fuel types (Diesel, Petrol) survived feature selection. The app offers an explicit "Other" option for everything else rather than implying coverage the model does not have.

## Run it locally

1. Download this repository (green **Code** button, then **Download ZIP**) and unzip it.
2. Open a terminal inside the unzipped folder.
3. Install the libraries: `pip install -r requirements.txt`
4. Start the app: `streamlit run app.py`

The app opens in your browser at `http://localhost:8501`.

## Project structure

```
vehicle-price-prediction/
├── README.md
├── app.py                                  # Streamlit app
├── car_price_prediction.ipynb              # full analysis and modelling
├── car_price_dataset.csv                   # dataset used by the notebook
├── requirements.txt
└── model/
    └── car_price_model_bundle.pkl          # trained model + scaler + preprocessing info
```

## Data

`car_price_dataset.csv` contains 15,411 used-car listings with 14 columns (brand, model, vehicle age, kilometres driven, mileage, engine size, maximum power, seats, fuel type, transmission, seller type and selling price in Rupees).

## Limitations and next steps

- Predictions for brands and fuel types outside the 3 + 2 listed are less precise. Target encoding would let the model use all brands.
- Performance on very high-priced vehicles is weaker. A separate valuation path or segment-level evaluation for premium cars is a natural next step.
- Only a modest hyperparameter search was run. A wider search, or an averaged ensemble of Random Forest and Gradient Boosting, could improve results.

## Tech stack

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Streamlit, joblib

## Note on the model file

The `.pkl` file is a Python pickle. Only load pickle files from sources you trust, and use the same scikit-learn version listed in `requirements.txt`.
