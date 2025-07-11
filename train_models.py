import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
import xgboost as xgb
import lightgbm as lgb
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
import joblib  
import os

# --- Configuration ---
DATA_PATH = "./csv_files/dat-merged.csv"
MODEL_SAVE_DIR = "saved_models" # Directory to save models and scaler

# Create the save directory if it doesn't exist
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Define file paths for saving
SCALER_PATH = os.path.join(MODEL_SAVE_DIR, "scaler.joblib")
COLUMNS_PATH = os.path.join(MODEL_SAVE_DIR, "columns.joblib")
XGB_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "xgboost_model.json")
LGB_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "lightgbm_model.txt")
TABNET_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "tabnet_model") # TabNet saves a zip implicitly

warnings.filterwarnings('ignore')

print(f"Starting model training process...")

# --- Data Loading and Initial Cleaning ---
print(f"Loading data from: {DATA_PATH}")
try:
    df = pd.read_csv(DATA_PATH, delimiter=',')
except FileNotFoundError:
    print(f"Error: Data file not found at '{DATA_PATH}'. Cannot proceed.")
    exit()

print("Initial data shape:", df.shape)
print("Cleaning data...")
df.drop(['name','preciptype','snow','snowdepth','windgust','severerisk','sunrise','sunset','moonphase','icon','stations','conditions','description'], axis=1, inplace=True)

df1 = df.copy()
df1['datetime'] = pd.to_datetime(df1['datetime'])
df1.set_index('datetime', inplace=True)

df1 = df1.iloc[:5479] # Limit data as in original script
print("Data shape after initial filtering:", df1.shape)

print("Handling missing values...")
missing_before = df1.isnull().sum()
# print("Missing values before handling:\n", missing_before[missing_before > 0])

df1['visibility'].fillna(4, inplace=True)
df1['sealevelpressure'].fillna(method='ffill', inplace=True)
df1.dropna(subset=['solarradiation', 'solarenergy'], inplace=True)

missing_after = df1.isnull().sum()
# print("\nMissing values after handling:\n", missing_after[missing_after > 0])
print("Data shape after handling missing values:", df1.shape)

# --- Feature Engineering and Scaling ---
all_numeric_columns = df1.select_dtypes(include=np.number).columns.tolist()
X = df1[all_numeric_columns].copy()
y = df1['temp'].copy()

print(f"Using features for scaling: {all_numeric_columns}")

scaler = StandardScaler()
print("Fitting scaler...")
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)

# Save the scaler and column lists
print(f"Saving scaler to {SCALER_PATH}")
joblib.dump(scaler, SCALER_PATH)

features_for_modeling = [col for col in all_numeric_columns if col != 'temp']
columns_to_save = {
    'all_numeric_columns': all_numeric_columns,
    'features_for_modeling': features_for_modeling
}
print(f"Saving column info to {COLUMNS_PATH}")
joblib.dump(columns_to_save, COLUMNS_PATH)
print(f"Features used for modeling: {features_for_modeling}")


# --- Train/Test Split ---
X_model_features = X_scaled_df[features_for_modeling]

X_train, X_test, y_train, y_test = train_test_split(
    X_model_features, y, test_size=0.2, random_state=42
)
print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# --- Model Definitions (Only those we need to save) ---
def create_xgboost_model():
    model = xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.01, max_depth=5, min_child_weight=1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, early_stopping_rounds=20
    )
    return model

def create_lightgbm_model():
    params = {
        'boosting_type': 'gbdt', 'objective': 'regression', 'metric': 'rmse',
        'num_leaves': 31, 'learning_rate': 0.01, 'feature_fraction': 0.9,
        'n_estimators': 1000, 'early_stopping_rounds': 20, 'verbose': -1
    }
    model = lgb.LGBMRegressor(**params)
    return model

def create_tabnet_model():
    model = TabNetRegressor(
        n_d=8, n_a=8, n_steps=3, gamma=1.3, n_independent=2, n_shared=2,
        lambda_sparse=1e-3, optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params=dict(mode="min", patience=10, min_lr=1e-5, factor=0.5),
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        mask_type='entmax', verbose=0
    )
    return model

# --- Training and Saving Models ---

# TabNet
print("\nTraining TabNet Model...")
tabnet_model = create_tabnet_model()
y_train_2d = y_train.values.reshape(-1, 1)
y_test_2d = y_test.values.reshape(-1, 1)
tabnet_model.fit(
    X_train=X_train.values, y_train=y_train_2d,
    eval_set=[(X_test.values, y_test_2d)],
    max_epochs=100, patience=20, batch_size=256,
    eval_metric=['mae']
)
# Evaluate (optional, just for confirmation during training)
tabnet_pred = tabnet_model.predict(X_test.values)
rmse_tabnet = np.sqrt(mean_squared_error(y_test, tabnet_pred))
r2_tabnet = r2_score(y_test, tabnet_pred)
print(f"TabNet Test Results -> RMSE: {rmse_tabnet:.4f}, R2: {r2_tabnet:.4f}")
print(f"Saving TabNet model to {TABNET_MODEL_PATH}.zip") # Note: .zip is added automatically
tabnet_model.save_model(TABNET_MODEL_PATH)


# XGBoost
print("\nTraining XGBoost Model...")
xgb_model = create_xgboost_model()
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

xgb_pred = xgb_model.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, xgb_pred))
r2_xgb = r2_score(y_test, xgb_pred)
print(f"XGBoost Test Results -> RMSE: {rmse_xgb:.4f}, R2: {r2_xgb:.4f}")
print(f"Saving XGBoost model to {XGB_MODEL_PATH}")
xgb_model.save_model(XGB_MODEL_PATH)


# LightGBM
print("\nTraining LightGBM Model...")
lgb_model = create_lightgbm_model()
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(20, verbose=False)]
)
# Evaluate (optional)
lgb_pred = lgb_model.predict(X_test)
rmse_lgb = np.sqrt(mean_squared_error(y_test, lgb_pred))
r2_lgb = r2_score(y_test, lgb_pred)
print(f"LightGBM Test Results -> RMSE: {rmse_lgb:.4f}, R2: {r2_lgb:.4f}")
print(f"Saving LightGBM model to {LGB_MODEL_PATH}")
lgb_model.booster_.save_model(LGB_MODEL_PATH) # Save the booster object

print("\n--- Model training and saving complete. ---")
