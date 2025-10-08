import pandas as pd
import joblib

def predict_new_data(test_path):
    model = joblib.load('models/churn_model.pkl')
    preprocessor = joblib.load('models/preprocessor.pkl')

    df = pd.read_csv(test_path)
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df['Subscription_Type'] = df['Subscription_Type'].map({'Basic': 0, 'Premium': 1})
    df['Last_Interaction_Type'] = df['Last_Interaction_Type'].map({'Neutral': 0, 'Negative': 1, 'Positive': 2})

    X = df.drop(columns=['Customer_ID'])
    X_processed = preprocessor.transform(X)
    churn_prob = model.predict_proba(X_processed)[:, 1]

    submission = pd.DataFrame({
        'Customer_ID': df['Customer_ID'],
        'Churn_Probability': churn_prob
    })

    submission.to_csv('submission.csv', index=False)
    print("✅ submission.csv generated successfully!")

if __name__ == "__main__":
    predict_new_data('data/test.csv')
