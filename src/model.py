import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_classifier(X, y, save_path="models/job_classifier.pkl"):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Classifier
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    # Save model to file
    joblib.dump(clf, save_path)

    return clf
