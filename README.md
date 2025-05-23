# Team Project 기계학습과데이터마이닝 (Machine Learning with Data Mining)

```
contributors: Marina, 명승주, 정윤교 and Aurèle
```

## First Idea:

We want to build a simple but solid model that predicts **whether it will rain or not** — kind of like how weather apps give you a percentage chance of rain. The idea comes from a Kaggle competition we found interesting:
🔗 [Kaggle Playground S5E3](https://www.kaggle.com/competitions/playground-series-s5e3/overview)

We’re given weather data (temperature, humidity, pressure, etc.) and the goal is to predict `rainfall > 0` — basically, will it rain or not?

## Step-by-step Plan:

1. **Preprocessing**

   * Clean the `.csv` dataset, handle missing values
   * Normalize features (StandardScaler etc.)
   * Maybe do some feature engineering if needed

2. **Model Training**

   * Try different models (Logistic Regression, Decision Tree, Random Forest, etc.)
   * Evaluate using accuracy, precision, F1-score
   * Pick the best one

3. **(Optional but cool) Real-world Extension**

   * Fetch real-time and historical data from Météo France (via API or scraping)
   * Build our own dataset (temperature, humidity, pressure + actual rainfall)
   * Compare our predictions to Météo France's past forecasts
   * See if our model sometimes does better 😉


---