import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def preprocess_data(train_path):
    df = pd.read_csv(train_path)

    for col in ['Gender', 'Subscription_Type', 'Last_Interaction_Type']:
        if not df[col].isnull().all():
            df[col].fillna(df[col].mode()[0], inplace=True)

    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df['Subscription_Type'] = df['Subscription_Type'].map({'Basic': 0, 'Premium': 1})
    df['Last_Interaction_Type'] = df['Last_Interaction_Type'].map({'Neutral': 0, 'Negative': 1, 'Positive': 2})

    X = df.drop(columns=['Customer_ID', 'Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_features = X.select_dtypes(include=['int64', 'float64']).columns
    cat_features = ['Location']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ]
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test, preprocessor
