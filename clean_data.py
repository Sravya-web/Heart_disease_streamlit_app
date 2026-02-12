import pandas as pd

df = pd.read_csv("data/health_activity_data.csv")

print("Shape:", df.shape)
print(df.head())
print(df.info())
print(df.isnull().sum())

df = df.drop_duplicates()
df.columns = df.columns.str.strip().str.replace(" ", "_")

df["Heart_Disease"] = df["Heart_Disease"].map({"Yes": 1, "No": 0})
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df["Smoker"] = df["Smoker"].map({"Yes": 1, "No": 0})
df["Diabetic"] = df["Diabetic"].map({"Yes": 1, "No": 0})

bp = df["Blood_Pressure"].astype(str).str.split("/", expand=True)
df["BP_Systolic"] = pd.to_numeric(bp[0], errors="coerce")
df["BP_Diastolic"] = pd.to_numeric(bp[1], errors="coerce")
df.drop("Blood_Pressure", axis=1, inplace=True)

num_cols = df.select_dtypes(include=["int64", "float64"]).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

cat_cols = df.select_dtypes(include=["object"]).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in ["Age", "BMI", "Daily_Steps", "Heart_Rate"]:
    if col in df.columns:
        df[col] = df[col].clip(lower=df[col].quantile(0.01), upper=df[col].quantile(0.99))

df.to_csv("data/health_activity_data_cleaned.csv", index=False)
print("Cleaned data saved to data/health_activity_data_cleaned.csv")

