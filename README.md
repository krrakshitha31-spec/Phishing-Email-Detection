# Phishing Email Detection Model

A machine learning-based phishing email detection system developed as part of a cybersecurity internship project.

The system analyzes email text and additional email characteristics to classify messages as **Phishing Email** or **Safe Email**.

## Features

* Trains a machine learning model using labeled phishing and legitimate emails
* Uses TF-IDF to analyze email text
* Detects URLs in email messages
* Counts suspicious keywords
* Analyzes special characters and exclamation marks
* Analyzes email length
* Classifies emails as Phishing or Safe
* Displays model accuracy
* Generates a confusion matrix
* Provides an interactive email checker
* Displays prediction confidence

## Dataset

The project uses a dataset containing **2,000 labeled emails**.

The dataset contains two categories:

* Safe Email
* Phishing Email

## Technologies Used

* Python
* Pandas
* Scikit-learn
* TF-IDF
* Logistic Regression
* Matplotlib
* Seaborn
* Regular Expressions

## Machine Learning Process

The system follows these steps:

1. Load the email dataset.
2. Split the dataset into training and testing data.
3. Convert email text into numerical features using TF-IDF.
4. Extract additional features such as URLs and suspicious keywords.
5. Combine the extracted features.
6. Train a Logistic Regression model.
7. Predict the email categories.
8. Calculate model accuracy.
9. Generate a confusion matrix.
10. Allow users to test individual emails interactively.

## Additional Email Features

The model extracts the following features from each email:

* Number of URLs
* Presence of a URL
* Number of suspicious keywords
* Number of special characters
* Number of exclamation marks
* Email length

## How to Run

### 1. Install Python

Make sure Python is installed on your system.

### 2. Install Required Libraries

Open a terminal in the project folder and run:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

### 3. Run the Program

```bash
python phishing_detector.py
```

### 4. Interactive Email Checker

After the model is trained, the program asks:

```text
Enter email text:
```

Paste or type an email message and press Enter.

The system will display:

```text
Prediction: Phishing Email
Confidence: XX.XX%
```

or:

```text
Prediction: Safe Email
Confidence: XX.XX%
```

Type `exit` to stop the interactive checker.

## Model Performance

The model achieved:

**100% accuracy** on the test dataset used during development.

The project also generates a confusion matrix named:

```text
confusion_matrix.png
```

## Project Structure

```text
Phishing-Email-Detection/
│
├── Phishing_validation_emails.csv
├── phishing_detector.py
├── confusion_matrix.png
└── README.md
```

## Security Note

This project is intended for educational and authorized cybersecurity purposes.

The model's performance depends on the dataset used for training and testing. High accuracy on this dataset does not guarantee perfect detection of all real-world phishing emails.
