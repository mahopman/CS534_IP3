import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef

models = {
    'Multi-layer Neural Network': (MLPClassifier(), {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
        'activation': ['relu', 'tanh', 'logistic', 'identity'],
        'learning_rate': ['constant', 'adaptive', 'invscaling'],
        'max_iter': [7000]  
    }),
    'Support Vector Machine': (SVC(), {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto', 0.1, 1]
    }),
    'K-Nearest Neighbors': (KNeighborsClassifier(), {
        'n_neighbors': [3, 5, 7],
        'p': [1, 2],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }),
    'Decision Tree': (DecisionTreeClassifier(), {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [None, 10, 20, 30],
        'ccp_alpha': [0.0, 0.01, 0.05]
    }),
    'Logistic Regression': (LogisticRegression(), {
        'penalty': ['l1', 'l2'],
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'saga'],
        'max_iter': [500]
    })
}

def load_and_preprocess_data(file_path):
    # load dataset
    df = pd.read_csv(file_path)

    # select the relevant columns
    df = df[['UDI', 'Product ID', 'Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Machine failure']]
    
    print("Step 3 DataFrame:")
    print(df)
    
    # one hot encode the 'Type' column
    df = pd.get_dummies(df, columns=['Type'], dtype=np.int64)
    
    # normalize the numerical columns between 0 and 1
    numerical_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    for numerical_col in numerical_cols:
        df[numerical_col] = (df[numerical_col] - df[numerical_col].min()) / (df[numerical_col].max() - df[numerical_col].min())
    
    print("\nStep 4 DataFrame:")
    print(df)
    
    return df

def undersample_data(df):
    # split the data into features(including key) and target
    features = ['Type_H', 'Type_L', 'Type_M', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    X = df[['UDI'] + features]
    y = df['Machine failure']
    
    # implement RandomUnderSampler
    X_resampled, y_resampled = RandomUnderSampler().fit_resample(X, y)

    # create a new DataFrame with resampled data
    resampled_df = pd.concat([X_resampled, y_resampled], axis=1).reset_index(drop=True)
    
    print("\nStep 5 DataFrame:")
    print(resampled_df)
    
    return resampled_df

def grid_search_cv(model, param_grid, X, y):
    # instantiate grid search with 5-fold cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='matthews_corrcoef', n_jobs=-1)
    # implement grid search
    grid_search.fit(X, y)
    # extract and return best params and scores from fit grid search
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    return best_params, best_score

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # grid search and cross-validation for each model
    results = {}
    for name, (model, param_grid) in models.items():
        best_params, best_score = grid_search_cv(model, param_grid, X_train, y_train)
        results[name] = {'params': best_params, 'score': best_score}
    
    print("\nTable 1: ML Model Performance")
    print("-" * 160)
    print("{:<30} {:<100} {:<20}".format("Model", "Best Parameter Values", "MCC-score on 5-fold CV"))
    print("-" * 160)
    for name, result in results.items():
        params_str = ', '.join(f"{k}={v}" for k, v in result['params'].items())
        print("{:<30} {:<100} {:<20.4f}".format(name, params_str, result['score']))
    print("-" * 160)
    
    # evaluate the best parameters for each model on the test set
    test_results = {}
    for name, (model, _) in models.items():
        model.set_params(**results[name]['params'])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mcc = matthews_corrcoef(y_test, y_pred)
        test_results[name] = {'params': results[name]['params'], 'score': mcc}
    
    print("\nTable 2: ML Model Performance on Test Set")
    print("-" * 160)
    print("{:<30} {:<100} {:<20}".format("Model", "Best Parameter Values", "MCC-score on Test Set"))
    print("-" * 160)
    for name, result in test_results.items():
        params_str = ', '.join(f"{k}={v}" for k, v in result['params'].items())
        print("{:<30} {:<100} {:<20.4f}".format(name, params_str, result['score']))
    print("-" * 160)
    
    best_model = max(test_results, key=lambda x: test_results[x]['score'])
    print(f"\nThe best model on the test set is {best_model} with a MCC score of {test_results[best_model]['score']:.4f}")

def main():
    # load and preprocess data
    df = load_and_preprocess_data('ai4i2020.csv')
    
    # re-balance data
    undersampled_data = undersample_data(df)
    
    # split the data into features and target
    features = ['Type_H', 'Type_L', 'Type_M', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    X = undersampled_data[features]
    y = undersampled_data['Machine failure']
    
    # split the data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # train and evaluate models
    train_and_evaluate_models(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()