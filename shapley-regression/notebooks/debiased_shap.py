import pickle

import lightgbm as lgb
import matplotlib.pyplot as plt
import shap
from shapreg import games, removal, shapley
from sklearn.model_selection import train_test_split

# Load data
X, y = shap.datasets.adult()
X_display, y_display = shap.datasets.adult(display=True)
num_features = X.shape[1]

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
d_train = lgb.Dataset(x_train, label=y_train)
d_tx_train = lgb.Dataset(
    x_train.loc[:, ~x_train.columns.isin(["Sex"])], label=x_train["Sex"]
)

d_tx_test = lgb.Dataset(
    x_test.loc[:, ~x_test.columns.isin(["Sex"])], label=x_test["Sex"]
)
d_test = lgb.Dataset(x_test, label=y_test)


params = {
    "max_bin": 512,
    "learning_rate": 0.05,
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 10,
    "verbose": -1,
    "min_data": 100,
    "boost_from_average": True,
}

tx_model = lgb.train(
    params,
    d_tx_train,
    10000,
    valid_sets=[d_tx_test],
    early_stopping_rounds=50,
    verbose_eval=1000,
)
# Train model


model = lgb.train(
    params,
    d_train,
    10000,
    valid_sets=[d_test],
    early_stopping_rounds=50,
    verbose_eval=1000,
)

# Make model callable
model_lam = lambda x: model.predict(x)
tx_model_lam = lambda x: tx_model.predict(x)
# Model extension
# marginal_extension = removal.MarginalExtension(x_test.values[:512], model_lam)

## Interventional
interventional_extension = removal.InterventionalExtension(
    x_test.values[:512], model_lam, tx_model_lam, 0.2
)
# Set up game (single prediction)
instance = X.values[0]

game = games.PredictionGame(interventional_extension, instance)

# Run estimator
explanation = shapley.ShapleyRegression(game, batch_size=32)

# Plot with 95% confidence intervals
feature_names = X.columns.tolist()
explanation.plot(feature_names, title="SHAP Values", sort_features=False)

plt.savefig("test.pdf")
