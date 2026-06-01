import seaborn as sns
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

df = sns.load_dataset('titanic')

df = df.drop(columns=['deck', 'alive'])
df = df.dropna()

X = df.drop(columns=['survived'])
y = df['survived']

cat_features = X.select_dtypes(include='object').columns.tolist()
num_features = X.select_dtypes(include=['number', 'bool']).columns.tolist()


cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore')),
])

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

preprocessor = ColumnTransformer([
    ('cat', cat_pipeline, cat_features),
    ('num', num_pipeline, num_features),
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42)),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

joblib.dump(pipeline, 'titanic_pipeline.pkl')
print("Model saved to titanic_pipeline.pkl")
