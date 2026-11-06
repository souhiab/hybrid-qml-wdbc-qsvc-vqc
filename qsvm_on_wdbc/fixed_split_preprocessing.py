"""Training-only preprocessing for the fixed-split simulator experiments."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def fixed_split(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def rfe_estimator(name):
    if name == "Logistic Regression":
        return LogisticRegression(
            penalty="l1", solver="saga", max_iter=5000, random_state=42
        )
    if name == "Random Forest":
        return RandomForestClassifier(random_state=42)
    raise ValueError(f"Unknown RFE estimator: {name}")


def prepare_features(X_train, X_test, y_train, scaling, estimator, k, *, raw=False):
    """Fit the scaler and RFE on training rows, then transform both partitions.

    With raw=True the scaler is used only for RFE, as in the original unscaled
    kernel comparison. The model receives the selected original columns.
    """
    scaler = {
        "Standardization": StandardScaler,
        "Normalization": MinMaxScaler,
    }[scaling]()
    X_train_scaled = scaler.fit_transform(X_train)
    selector = RFE(rfe_estimator(estimator), n_features_to_select=k, step=1)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    names = X_train.columns[selector.support_].tolist()
    if raw:
        return X_train[names].to_numpy(), X_test[names].to_numpy(), names

    X_test_selected = selector.transform(scaler.transform(X_test))
    return X_train_selected, X_test_selected, names
