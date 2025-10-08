import joblib
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from preprocess import preprocess_data

def train_model():
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data('data/train.csv')

    base_model = AdaBoostClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500, 700],
        'learning_rate': [0.1, 0.3, 0.5]
    }

    grid = GridSearchCV(base_model, param_grid, scoring='f1', cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(best_model, 'models/churn_model.pkl')
    joblib.dump(preprocessor, 'models/preprocessor.pkl')

    print(f"Best params: {grid.best_params_}")
    print("Model and preprocessor saved successfully!")

if __name__ == "__main__":
    train_model()
