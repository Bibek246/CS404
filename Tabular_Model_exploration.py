import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Load the dataset
df = pd.read_csv("fuel_end_use.csv")

# Define outcomes and features
outcomes = ['Process Heating', 'CHP and/or Cogeneration Process', 'Conventional Boiler Use']
features = ["Coal", "Diesel", "Natural_gas", "Other", "Residual_fuel_oil", "Temp_degC", "Total"]

# Take a random sample of 30,000 rows
df = df.sample(n=30000)

# Convert to numeric
for feature in features:
    df[feature] = pd.to_numeric(df[feature], errors="coerce")

# Drop rows with missing values
df = df.dropna()

# Take a look at the data by end-use
ph_df = df.loc[df["END_USE"] == outcomes[0]]
chp_df = df.loc[df["END_USE"] == outcomes[1]]
boil_df = df.loc[df["END_USE"] == outcomes[2]]

# Function to visualize boxplots
def plot_var(variable):
    plt.boxplot([ph_df[variable], chp_df[variable], boil_df[variable]])
    plt.title(variable)
    plt.show()
    plt.clf()

# Example of calling plot_var function to visualize a feature
# plot_var("Coal")

# Function to prepare the dataset for model training and testing
def prep_dataset(df, features, outcome):
    feature_df = df[features]
    outcome_df = df[outcome]
    
    # Encode labels
    encoder = LabelEncoder()
    outcome_df = encoder.fit_transform(outcome_df)
    
    print("Classes:", encoder.classes_)
    
    # Normalize the data
    scaler = MinMaxScaler()
    feature_df = scaler.fit_transform(feature_df)
    
    # Split into a training and test set
    train_in, test_in, train_out, test_out = train_test_split(feature_df, outcome_df, test_size=0.2, random_state=42)
    return train_in, test_in, train_out, test_out

# Prepare dataset
train_in, test_in, train_out, test_out = prep_dataset(df, features, "END_USE")

# Function to run a model and calculate metrics
def run_model(model, train_in, test_in, train_out, test_out):
    model.fit(train_in, train_out)
    test_accuracy_score = model.score(test_in, test_out)
    training_accuracy_score = model.score(train_in, train_out)
    
    print(f"Training accuracy: {training_accuracy_score}")
    print(f"Test accuracy: {test_accuracy_score}")
    
    # Per-class accuracy (F1 score)
    test_predictions = model.predict(test_in)
    f1_per_class = f1_score(test_out, test_predictions, average=None)
    print(f"F1 Score per class: {f1_per_class}")

# Define models with different hyperparameter settings
models = [
    # Decision Tree with different depths and criterion
    DecisionTreeClassifier(max_depth=10),
    DecisionTreeClassifier(max_depth=20),
    DecisionTreeClassifier(criterion="entropy"),

    # Random Forest with different n_estimators and depth
    RandomForestClassifier(n_estimators=50, max_depth=10),
    RandomForestClassifier(n_estimators=100, max_depth=20),
    RandomForestClassifier(n_estimators=200, criterion="entropy"),

    # SVM with different kernels
    SVC(kernel="linear"),
    SVC(kernel="rbf"),
    SVC(kernel="poly", degree=3)
]

# Run each model and display the results
for i, model in enumerate(models, 1):
    print(f"\nRunning Model {i}: {model}")
    run_model(model, train_in, test_in, train_out, test_out)
