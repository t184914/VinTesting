import joblib, pandas as pd

d = joblib.load("output/best_model.joblib")
imp = pd.Series(d["model"].feature_importances_, index=d["features"])
print(imp.sort_values(ascending=False))