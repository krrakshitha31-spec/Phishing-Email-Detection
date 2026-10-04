import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from scipy.sparse import hstack, csr_matrix

import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------------------
# 1. Load the dataset
# -----------------------------------------

data = pd.read_csv("Phishing_validation_emails.csv")

print("Dataset loaded successfully!")
print("Number of emails:", len(data))

print("\nDataset columns:")
print(data.columns.tolist())


# -----------------------------------------
# 2. Prepare the data
# -----------------------------------------

X = data["Email Text"].fillna("")
y = data["Email Type"]


# -----------------------------------------
# 3. Split the dataset
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -----------------------------------------
# 4. Convert email text into TF-IDF features
# -----------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# -----------------------------------------
# 5. Define suspicious keywords
# -----------------------------------------

suspicious_keywords = [
    "urgent",
    "verify",
    "verification",
    "account",
    "password",
    "login",
    "click",
    "suspended",
    "confirm",
    "security alert",
    "limited",
    "expire",
    "payment",
    "bank",
    "credit card",
    "reset",
    "winner",
    "prize"
]


# -----------------------------------------
# 6. Extract additional email features
# -----------------------------------------

def extract_features(emails):

    features = []

    for email in emails:

        email = str(email)

        # Count URLs
        urls = re.findall(
            r'https?://\S+|www\.\S+',
            email,
            re.IGNORECASE
        )

        url_count = len(urls)

        # Check whether email contains a URL
        has_url = 1 if url_count > 0 else 0

        # Count suspicious keywords
        keyword_count = 0

        email_lower = email.lower()

        for keyword in suspicious_keywords:

            if keyword in email_lower:
                keyword_count += 1

        # Count special characters
        special_char_count = len(
            re.findall(r'[^a-zA-Z0-9\s]', email)
        )

        # Count exclamation marks
        exclamation_count = email.count("!")

        # Calculate email length
        email_length = len(email)

        features.append([
            url_count,
            has_url,
            keyword_count,
            special_char_count,
            exclamation_count,
            email_length
        ])

    return features


X_train_features = extract_features(X_train)
X_test_features = extract_features(X_test)


# -----------------------------------------
# 7. Scale additional features
# -----------------------------------------

scaler = StandardScaler()

X_train_features = scaler.fit_transform(
    X_train_features
)

X_test_features = scaler.transform(
    X_test_features
)

X_train_features = csr_matrix(
    X_train_features
)

X_test_features = csr_matrix(
    X_test_features
)


# -----------------------------------------
# 8. Combine all features
# -----------------------------------------

X_train_combined = hstack([
    X_train_tfidf,
    X_train_features
])

X_test_combined = hstack([
    X_test_tfidf,
    X_test_features
])


print("\nAdditional email features:")
print("1. URL count")
print("2. URL presence")
print("3. Suspicious keyword count")
print("4. Special character count")
print("5. Exclamation mark count")
print("6. Email length")


# -----------------------------------------
# 9. Train the machine learning model
# -----------------------------------------

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_combined,
    y_train
)


# -----------------------------------------
# 10. Make predictions
# -----------------------------------------

y_pred = model.predict(
    X_test_combined
)


# -----------------------------------------
# 11. Calculate accuracy
# -----------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n" + "=" * 50)
print("PHISHING EMAIL DETECTION MODEL")
print("=" * 50)

print(f"\nTraining emails: {len(X_train)}")
print(f"Testing emails: {len(X_test)}")

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")


# -----------------------------------------
# 12. Display confusion matrix
# -----------------------------------------

labels = [
    "Phishing Email",
    "Safe Email"
]

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print("\nConfusion Matrix:")
print(cm)


# -----------------------------------------
# 13. Plot confusion matrix
# -----------------------------------------

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.title(
    "Phishing Email Detection - Confusion Matrix"
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png"
)

plt.show()

print(
    "\nConfusion matrix saved as: confusion_matrix.png"
)


# -----------------------------------------
# 14. Interactive Email Checker
# -----------------------------------------

print("\n" + "=" * 50)
print("INTERACTIVE PHISHING EMAIL CHECKER")
print("=" * 50)

print("\nPaste an email below.")
print("Type 'exit' to stop the checker.")

while True:

    print("\n" + "-" * 50)

    email_input = input(
        "\nEnter email text: "
    )

    # Stop the program
    if email_input.lower() == "exit":
        print("\nEmail checker stopped.")
        break

    # Check empty input
    if not email_input.strip():
        print("\nPlease enter an email.")
        continue

    # Convert email text into TF-IDF features
    email_tfidf = vectorizer.transform(
        [email_input]
    )

    # Extract additional features
    email_features = extract_features(
        [email_input]
    )

    # Scale the features using the trained scaler
    email_features = scaler.transform(
        email_features
    )

    # Convert to sparse format
    email_features = csr_matrix(
        email_features
    )

    # Combine TF-IDF and additional features
    email_combined = hstack([
        email_tfidf,
        email_features
    ])

    # Make prediction
    prediction = model.predict(
        email_combined
    )[0]

    # Get prediction probability
    probabilities = model.predict_proba(
        email_combined
    )[0]

    confidence = max(probabilities) * 100

    print("\nPrediction:", prediction)
    print(f"Confidence: {confidence:.2f}%")

    if prediction == "Phishing Email":

        print(
            "Warning: This email may be a phishing attempt."
        )

    else:

        print(
            "This email appears to be safe based on the model."
        )
