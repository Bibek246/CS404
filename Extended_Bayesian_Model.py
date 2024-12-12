import pandas as pd
import matplotlib as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import seaborn as sns

# Load the dataset
df = pd.read_csv("faults.csv")
print(df.head())


# Fault labels
faults = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]

# Create a new 'fault' column based on these fault labels
df['fault'] = 0

# Assign integer values for each fault type
for i in range(len(faults)):
    true_fault_indexes = df.loc[df[faults[i]] == 1].index.tolist()
    df.loc[true_fault_indexes, 'fault'] = i + 1

# Define outcomes (the fault column)
outcomes = df['fault']

# Experiment with different subsets of features
# First subset: We'll select features that are related to physical measurements
subset1_features = ['X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas', 
                    'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity']

# Train and test with the first subset
train_features_1, test_features_1, train_outcomes_1, test_outcomes_1 = train_test_split(
    df[subset1_features], outcomes, test_size=0.10, random_state=42 #to ensure reproducibility of results
)

bayes_classifier_1 = GaussianNB()
bayes_classifier_1.fit(train_features_1, train_outcomes_1)

# Evaluate the model
accuracy_1 = bayes_classifier_1.score(test_features_1, test_outcomes_1)

# Second subset: We'll include features related to luminosity and steel type
subset2_features = ['Sum_of_Luminosity', 'Minimum_of_Luminosity', 'Maximum_of_Luminosity', 
                    'TypeOfSteel_A300', 'TypeOfSteel_A400', 'Steel_Plate_Thickness']

# Train and test with the second subset
train_features_2, test_features_2, train_outcomes_2, test_outcomes_2 = train_test_split(
    df[subset2_features], outcomes, test_size=0.10, random_state=42
)

bayes_classifier_2 = GaussianNB()
bayes_classifier_2.fit(train_features_2, train_outcomes_2)

# Evaluate the second model
accuracy_2 = bayes_classifier_2.score(test_features_2, test_outcomes_2)

# Now, let's test with 5% and 20% test sizes on the first subset of features
# First with a 5% test set
train_features_5pct, test_features_5pct, train_outcomes_5pct, test_outcomes_5pct = train_test_split(
    df[subset1_features], outcomes, test_size=0.05, random_state=42
)
bayes_classifier_1_5pct = GaussianNB()
bayes_classifier_1_5pct.fit(train_features_5pct, train_outcomes_5pct)
accuracy_5pct = bayes_classifier_1_5pct.score(test_features_5pct, test_outcomes_5pct)

# Now with a 20% test set
train_features_20pct, test_features_20pct, train_outcomes_20pct, test_outcomes_20pct = train_test_split(
    df[subset1_features], outcomes, test_size=0.20, random_state=42
)
bayes_classifier_1_20pct = GaussianNB()
bayes_classifier_1_20pct.fit(train_features_20pct, train_outcomes_20pct)
accuracy_20pct = bayes_classifier_1_20pct.score(test_features_20pct, test_outcomes_20pct)

# Best model analysis: Get classification report for subset 1 (best so far)
predictions = bayes_classifier_1.predict(test_features_1)
classification_report_result = classification_report(test_outcomes_1, predictions, target_names=faults, zero_division=1)

# Print the results
print("Accuracy for Subset 1 (Physical Measurements):", accuracy_1)
print("Accuracy for Subset 2 (Luminosity and Steel Type):", accuracy_2)
print("Accuracy with 5% test set (Subset 1):", accuracy_5pct)
print("Accuracy with 20% test set (Subset 1):", accuracy_20pct)
print("\nClassification Report for Best Model (Subset 1):\n", classification_report_result)


