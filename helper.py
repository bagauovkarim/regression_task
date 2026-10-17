import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import FunctionTransformer

LEAK_COLS = ["annual_premium", "monthly_premium"]
ID_COLS = ["person_id"]
SKEWED_COLS = ["hba1c", "income", "total_claims_paid", "avg_claim_amount"]

def load_clean_data(path="medical_insurance.csv"):
    
    df = pd.read_csv(path)

    df["alcohol_freq"] = df["alcohol_freq"].fillna("None") 

    # удаляем утечки и идентификатор
    df = df.drop(columns=LEAK_COLS + ID_COLS)

    # дубликатов в датасете нет, но на всякий случай
    df = df.drop_duplicates()

    return df


def get_preprocessor(X_train):
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()

    ord_cols = ["education", "smoker", "alcohol_freq", "network_tier"] # порядковые
    onehot_cols = [c for c in cat_cols if c not in ord_cols] # остальные - one hot

    #Числовые
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    # Теперь все нужно собрать в один ColumnTransformer через мини Pipeline'ы 

    ohe_pipe = Pipeline([
        ('ohe', OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    ord_pipe = Pipeline([
        ('ord', OrdinalEncoder(categories=[
            ["No HS", "HS", "Some College", "Bachelors", "Masters", "Doctorate"], #education
            ["Never", "Former", "Current"], #smoker
            ["None", "Occasional", "Weekly", "Daily"], #alcohol_freq
            ["Bronze", "Silver", "Gold", "Platinum"], #network_tier
        ]))
    ])

    preprocessor = ColumnTransformer([
        ('num', 'passthrough', num_cols),
        ('ohe', ohe_pipe, onehot_cols),
        ('ord', ord_pipe, ord_cols)
    ])
    return preprocessor


def evaluate_models_cv(models, X, y, preprocessor, cv, scoring=None):
    
    if scoring is None:
        scoring = ["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"]

    results = {}
    for name, model in models:
        pipe = Pipeline([
            ("prep", clone(preprocessor)),
            ("model", clone(model)),
        ])
        cv_res = cross_validate(pipe, X, y, cv=cv, scoring=scoring)
        row = {}
        for m in scoring:
            score = cv_res["test_" + m].mean()
            if m.startswith("neg_"):
                row[m.replace("neg_", "")] = -score   # RMSE, MAE — положительные
            else:
                row[m] = score
        results[name] = row

    return pd.DataFrame(results).T

def get_preprocessor_scaled(X_train):

    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    
    ord_cols = ["education", "smoker", "alcohol_freq", "network_tier"] # порядковые
    onehot_cols = [c for c in cat_cols if c not in ord_cols] # остальные - one hot
    
    #Числовые
    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    # Теперь все нужно собрать в один ColumnTransformer через мини Pipeline'ы

    ohe_pipe = Pipeline([
        ('ohe', OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
   
    ord_pipe = Pipeline([
        ('ord', OrdinalEncoder(categories=[
            ["No HS", "HS", "Some College", "Bachelors", "Masters", "Doctorate"], #education
            ["Never", "Former", "Current"], #smoker
            ["None", "Occasional", "Weekly", "Daily"], #alcohol_freq
            ["Bronze", "Silver", "Gold", "Platinum"], #network_tier
        ]))
    ])
   
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('ohe', ohe_pipe, onehot_cols),
        ('ord', ord_pipe, ord_cols)
    ])

    return preprocessor


def get_preprocessor_log(X_train):
    cat_cols = X_train.select_dtypes("object").columns.tolist()
    ord_cols = ["education", "smoker", "alcohol_freq", "network_tier"]
    onehot_cols = [c for c in cat_cols if c not in ord_cols]
    num_cols = X_train.select_dtypes("number").columns.tolist()

    # делим числовые: под лог и остальные
    skewed    = [c for c in num_cols if c in SKEWED_COLS]
    other_num = [c for c in num_cols if c not in SKEWED_COLS]

    ord_pipe = Pipeline([
        ("ord", OrdinalEncoder(categories=[
            ["No HS", "HS", "Some College", "Bachelors", "Masters", "Doctorate"],
            ["Never", "Former", "Current"],
            ["None", "Occasional", "Weekly", "Daily"],
            ["Bronze", "Silver", "Gold", "Platinum"],
        ], handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocessor = ColumnTransformer([
        ("num",  "passthrough", other_num),
        ("log",  FunctionTransformer(np.log1p), skewed),  
        ("ohe",  OneHotEncoder(handle_unknown="ignore", sparse_output=False), onehot_cols),
        ("ord",  ord_pipe, ord_cols),
      ])
    return preprocessor