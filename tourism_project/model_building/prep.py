import pandas as pd
from sklearn.model_selection import train_test_split

#Load dataset from data folder
df = pd.read_csv("tourism_project/data/tourism.csv")

# Drop columns with no header (empty string or NaN in column names)
df = df.drop(columns=[col for col in df.columns if col.strip() == "" or pd.isna(col)])

# Drop column CustomerID since it has only unique values
df.drop(columns=["CustomerID"], inplace=True)

# Standardize gender column
df['Gender'] = df['Gender'].replace({'Fe Male': 'Female'})

# NOTE: Categorical variables are left unchanged.
# The training pipeline one-hot-encodes it, and the Streamlit app also sends
# raw values. Encoding it here (e.g. LabelEncoder) would make training
# and serving use different representations, silently breaking predictions.

#Drop target variable ProdTaken from X
X = df.drop(columns=["ProdTaken"])
#Add target variable ProdTaken to Y
y = df["ProdTaken"]

#Split dataset into training and testing sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Save datasets locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
