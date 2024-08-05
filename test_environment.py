import pandas as pd
print("running")
try:
    y_train = pd.read_pickle('data/feature/train_target.pkl')
    print("Loaded successfully:", y_train.head())
except Exception as e:
    print("Error loading train_target.pkl:", e)

