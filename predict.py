import os
import joblib
import pandas as pd

_MODEL = None
_FEATURES = None


def _locate_sample_csv():
    candidates = [
        "data/health_activity_data_cleaned.csv",
        "data/health_activity_data.csv",
        "Data/health_activity_data.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_model(path="heart_disease_model.pkl"):
    global _MODEL, _FEATURES
    if _MODEL is None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        _MODEL = joblib.load(path)

        # determine expected feature columns from a sample CSV if available
        sample_csv = _locate_sample_csv()
        if sample_csv:
            df = pd.read_csv(sample_csv)
            if "Heart_Disease" in df.columns:
                _FEATURES = [c for c in df.columns if c != "Heart_Disease"]
            else:
                _FEATURES = list(df.columns)
        # if original CSV used a combined Blood_Pressure column, replace it
        if _FEATURES and "Blood_Pressure" in _FEATURES:
            i = _FEATURES.index("Blood_Pressure")
            _FEATURES.pop(i)
            _FEATURES.insert(i, "BP_Diastolic")
            _FEATURES.insert(i, "BP_Systolic")
        else:
            _FEATURES = None
    return _MODEL


def _prepare_input(input_dict):
    # Build a single-row DataFrame for prediction. Try to keep columns in same order
    # as training data when possible using the sample CSV.
    sample_csv = _locate_sample_csv()
    if sample_csv:
        sample = pd.read_csv(sample_csv, nrows=1)
        cols = list(sample.columns)
        if "Heart_Disease" in cols:
            cols.remove("Heart_Disease")
        # training code split Blood_Pressure into BP_Systolic and BP_Diastolic
        if "Blood_Pressure" in cols:
            # replace Blood_Pressure placeholder with two columns
            i = cols.index("Blood_Pressure")
            cols.pop(i)
            cols.insert(i, "BP_Diastolic")
            cols.insert(i, "BP_Systolic")
    else:
        cols = list(input_dict.keys())

    row = {}
    # handle blood pressure if provided as combined or split
    if "Blood_Pressure" in input_dict and ("BP_Systolic" not in input_dict and "BP_Diastolic" not in input_dict):
        try:
            s, d = str(input_dict["Blood_Pressure"]).split("/")
            input_dict["BP_Systolic"] = int(s)
            input_dict["BP_Diastolic"] = int(d)
        except Exception:
            pass

    for c in cols:
        if c in input_dict:
            row[c] = input_dict[c]
        else:
            # fill with NaN for missing cols; model should handle numeric NaNs
            row[c] = pd.NA

    df = pd.DataFrame([row])

    # Map common categorical values to numeric as used during training
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0}).fillna(df["Gender"])
    if "Smoker" in df.columns:
        df["Smoker"] = df["Smoker"].map({"Yes": 1, "No": 0}).fillna(df["Smoker"])
    if "Diabetic" in df.columns:
        df["Diabetic"] = df["Diabetic"].map({"Yes": 1, "No": 0}).fillna(df["Diabetic"])

    # Ensure numeric types where possible
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # If we discovered feature order earlier, reorder columns to that order
    global _FEATURES
    if _FEATURES:
        # keep only features that exist in df
        ordered = [c for c in _FEATURES if c in df.columns]
        df = df.reindex(columns=ordered)

    return df


def predict_single(input_dict, model_path="heart_disease_model.pkl"):
    model = load_model(model_path)
    X = _prepare_input(input_dict)
    # fill numerical NaNs with 0 (conservative). In real deployment, mirror training preprocessing.
    X_numeric = X.select_dtypes(include=["number"]) 
    X[X_numeric.columns] = X_numeric.fillna(0)

    pred_proba = None
    pred = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        pred_proba = float(proba[0][1]) if proba.shape[1] > 1 else float(proba[0][0])
    pred = int(model.predict(X)[0])

    return {"prediction": pred, "probability": pred_proba, "input": X.to_dict(orient="records")[0]}


if __name__ == "__main__":
    # quick local test
    load_model()
    example = {"Age": 60, "Gender": "Male", "BMI": 30, "Daily_Steps": 5000, "Heart_Rate": 80, "BP_Systolic": 130, "BP_Diastolic": 80, "Smoker": "No", "Diabetic": "No"}
    print(predict_single(example))
