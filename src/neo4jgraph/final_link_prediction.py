import csv
import sqlite3
import joblib
import pandas as pd
from neo4j import GraphDatabase
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

############################ Final VALIDATION ###################################

# Connect to Neo4J
uri = "bolt://localhost:7002"
driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))

# Get all positive links between websites and stakeholders and stakeholders to stakeholders
with driver.session(database="neo4j") as session:
    result_positive = session.run(f"""
        MATCH (second_website)-[rel:SLOT_2024]->(stakeholder_pers)
        RETURN id(second_website) AS node1, id(stakeholder_pers) AS node2, 1 AS label
        UNION
        MATCH (second_website)-[rel:SLOT_2024]->(stakeholder_org)
        RETURN id(second_website) AS node1, id(stakeholder_org) AS node2, 1 AS label
        UNION
        MATCH (stakeholder_org)-[rel:SLOT_2024]->(stakeholder_pers)
        RETURN id(stakeholder_org) AS node1, id(stakeholder_pers) AS node2, 1 AS label
    """)
    # Store node pair table in dataframe and csv
    existing_links = pd.DataFrame([dict(record) for record in result_positive])
    existing_links = existing_links.drop_duplicates()
    existing_links.to_csv(f'df_split_2024_existing_links.csv', index=False)

# Apply feature engineering to the positive links
def apply_graphy_features(data, rel_type):
    uri = "bolt://localhost:7002"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))

    query = """
    UNWIND $pairs AS pair
    MATCH (p1) WHERE id(p1) = pair.node1
    MATCH (p2) WHERE id(p2) = pair.node2
    RETURN pair.node1 AS node1,
           pair.node2 AS node2,
           gds.alpha.linkprediction.commonNeighbors(p1, p2, {
             relationshipQuery: $relType}) AS cn,
           gds.alpha.linkprediction.preferentialAttachment(p1, p2, {
             relationshipQuery: $relType}) AS pa,
           gds.alpha.linkprediction.adamicAdar(p1, p2, {
             relationshipQuery: $relType}) AS aa,
           gds.alpha.linkprediction.totalNeighbors(p1, p2, {
             relationshipQuery: $relType}) AS tn,
           gds.alpha.linkprediction.resourceAllocation(p1, p2, {
             relationshipQuery: $relType}) AS ra
    """
    # Calculate per node pair
    pairs = [{"node1": node1, "node2": node2} for node1, node2 in data[["node1", "node2"]].values.tolist()]

    # Query the feature values per node pair and store in dataframe
    with driver.session(database="neo4j") as session:
        result = session.run(query, {"pairs": pairs, "relType": rel_type})
        features = pd.DataFrame([dict(record) for record in result])
    return pd.merge(data, features, on = ["node1", "node2"])

# Read node to node csv and insert the year created
df_split_year = pd.read_csv(f'df_split_2024_existing_links.csv')
df_split_year.insert(0, "year", '2024-01-01 00:00:00')

# Execute the query
df_results = apply_graphy_features(df_split_year, f"SLOT_2024")
print(df_results.sample(5, random_state=42))

# Store the year split in a csv and in sqllite database
df_results.to_csv(f'df_split_2024_existing_links_2.csv', index=False)

# Connect to the SQL database "final_predictions" that contains the predictions of 2024 and 2025
conn = sqlite3.connect('final_prediction.db')
cursor = conn.cursor()

# Execute SQL query to select rows where "year" column equals "2024-01-01 00:00:00" to compare to actual 2024 links
# for validation
query = "SELECT * FROM 'final prediction' WHERE year = '2024-01-01 00:00:00'"
cursor.execute(query)
rows = cursor.fetchall()


output_csv_file = "2024_ts_data.csv"

# Write 2024 dataset to CSV file
with open(output_csv_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    column_names = ['year', 'node1', 'node2', 'label', 'cn', 'pa', 'aa', 'tn', 'ra']
    csv_writer.writerow(column_names)
    csv_writer.writerows(rows)
    print("Modified DataFrame saved to '2024_ts_data.csv'")
conn.close()

# Concatenate both datasets to one dataset
ts_2024 = pd.read_csv(f'2024_ts_data.csv')
existing_links_2024 = pd.read_csv(f'df_split_2024_existing_links_2.csv')
final_dataset = pd.concat([ts_2024, existing_links_2024])

# Shuffle the rows and save the shuffled dataset to a csv file
final_2024_val_hybrid = final_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
final_2024_val_hybrid.to_csv('final_2024_val_hybrid.csv', index=False)

# load best classifier model (XGBoost slot4)
classifier = joblib.load('model_slot_4.pkl')

columns = [
    "cn", "pa", "aa", "tn", "ra"
]

# Make predictions with actual labels as test labels
predictions = classifier.predict(final_2024_val_hybrid[columns])
y_test = final_2024_val_hybrid["label"]

# Evaluate the model for accuracy, precision, recall, etc.
def evaluate_model(predictions, actual):
    actual = np.array(actual)
    return pd.DataFrame({
        "Measure": ["Accuracy", "Precision", "Recall"],
        "Score": [accuracy_score(actual, predictions),
                  precision_score(actual, predictions),
                  recall_score(actual, predictions)]
    })

print("-" * 55)
print(f'Evaluation split Validaton:')
print(evaluate_model(predictions, y_test))

# We do not only want to compute the general accuracy, but want to split it class wise, since our data is imbalanced
def evaluate_model_class(predictions, actual):
    report = classification_report(actual, predictions, target_names=['Class 0', 'Class 1'], output_dict=True)

    # Calculate confusion matrix to get the number of true positives
    cm = confusion_matrix(actual, predictions)
    # Extract True Positives  for each class
    tp_class_0 = cm[0, 0]  # True positives for class 0
    fn_class_0 = cm[0, 1]
    fp_class_0 = cm[1, 0]
    tp_class_1 = cm[1, 1]  # True positives for class 1
    fn_class_1 = cm[1, 0]
    fp_class_1 = cm[0, 1]

    # Add true positives and total actual instances to the report
    report['Class 0']['true_positives'] = tp_class_0
    report['Class 0']['total_actual'] = report['Class 0']['support']  # Total actual instances for class 0
    report['Class 0']['false_positives'] = fp_class_0
    report['Class 0']['false_negatives'] = fn_class_0
    report['Class 1']['true_positives'] = tp_class_1
    report['Class 1']['total_actual'] = report['Class 1']['support']  # Total actual instances for class 1
    report['Class 1']['false_positives'] = fp_class_1
    report['Class 1']['false_negatives'] = fn_class_1

    return report, cm

# make the prediction
report, cm = evaluate_model_class(predictions, y_test)

# Print out the results
print("-" * 55)
counter = 0
for class_label, metrics in report.items():
    if isinstance(metrics, dict):
        counter += 1
        print(f"Metrics for {class_label} in split VAL:")
        print(f"Precision: {metrics['precision']}")
        print(f"Recall: {metrics['recall']}")
        print(f"F1-score: {metrics['f1-score']}")
        print(f"Actual positives: {metrics['support']}")
        print(f"True positives: {metrics['true_positives']}")
        print(f"False positives: {metrics['false_positives']}")
        print(f"False negatives: {metrics['false_negatives']}")
        print("-" * 55)
        if counter == 2:
            break

# Look into the Top 30 predictions:
predicted_probabilities = classifier.predict_proba(final_2024_val_hybrid[columns])
# Extract probabilities for class 1
class_1_probabilities = predicted_probabilities[:, 1]
# Sort indices based on probabilities in descending order
sorted_indices = np.argsort(class_1_probabilities)[::-1]
# Select the top 30 indices and extract corresponding node pairs
top_thirty_indices = sorted_indices[:30]
top_thirty_node_pairs = final_2024_val_hybrid.iloc[top_thirty_indices]

# Evaluate precision, recall, and accuracy on actual data for these node pairs
actual_data = final_2024_val_hybrid["label"].iloc[top_thirty_indices]
predictions_for_top_thirty = classifier.predict(final_2024_val_hybrid[columns].iloc[top_thirty_indices])

# Compute accuracy, precision and recall
precision = precision_score(actual_data, predictions_for_top_thirty)
recall = recall_score(actual_data, predictions_for_top_thirty)
accuracy = accuracy_score(actual_data, predictions_for_top_thirty)

print("-" * 55)
print("Precision top 30:", precision)
print("Recall top 30:", recall)
print("Accuracy top 30:", accuracy)
print("-" * 55)

# Get the first 5 of them:
top_five_node_pairs = final_2024_val_hybrid.iloc[top_thirty_indices[:5]]
top_five_probabilities = class_1_probabilities[top_thirty_indices[:5]]
top_five_actual_class = actual_data.iloc[:5]

# Print the top 5 node pairs
print("Top 5 Node Pairs:")
for i, (index, row) in enumerate(top_five_node_pairs.iterrows()):
    print(f"Node Pair {i + 1}:")
    print("Prediction Probability:", top_five_probabilities[i])
    print("Actual Class:", top_five_actual_class.iloc[i])
    print(row)  # Assuming each row represents a node pair
    print()



# Compute the AUROC curve

# Compute probability estimates for the positive class
probs = classifier.predict_proba(final_2024_val_hybrid[columns])[:, 1]

# Compute AUROC curve
fpr, tpr, thresholds = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)

# Plot AUROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Area Under Curve (AUC) for slot VAL')
plt.legend(loc="lower right")
plt.show()
print("AUC:", roc_auc)

# Compute AUPRC which is a better indicator for imbalanced classes
# Compute precision and recall
precision, recall, _ = precision_recall_curve(y_test, probs)
auprc = auc(recall, precision)

# Plot AUPRC curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve (AUPRC = %0.2f)' % auprc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Area Under Precision-Recall Curve (AUPRC) for slot VAL')
plt.legend(loc="lower left")
plt.show()
print("AUPRC:", auprc)











############################ Final PREDICTION ####################################
# The test set of 2025 is the test set of 2024 + the changes reported in time-series predictions
conn = sqlite3.connect('final_prediction.db')
cursor = conn.cursor()

# Execute SQL query to select rows where "year" column equals "2025-01-01 00:00:00" to get the prediction for 2025
query = "SELECT * FROM 'final prediction' WHERE year = '2025-01-01 00:00:00'"
cursor.execute(query)
rows = cursor.fetchall()

# Write fetched data to CSV file
output_csv_file = "2025_final_data.csv"
with open(output_csv_file, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    column_names = ['year', 'node1', 'node2', 'label', 'cn', 'pa', 'aa', 'tn', 'ra']
    csv_writer.writerow(column_names)
    csv_writer.writerows(rows)
conn.close()

print("Modified DataFrame saved to '2025_final_data.csv'")
final_2025 = pd.read_csv(f'2025_final_data.csv')



# load best classifier model (XGBoost for slot 4)
classifier = joblib.load('model_slot_4.pkl')

columns = [
    "cn", "pa", "aa", "tn", "ra"
]

# Make predictions on forecasted values
predictions = classifier.predict(final_2025[columns])
print("-" * 55)

# Look into the Top 30 predictions:
predicted_probabilities = classifier.predict_proba(final_2025[columns])
# Extract probabilities for class 1
class_1_probabilities = predicted_probabilities[:, 1]
# Sort indices based on probabilities in descending order
sorted_indices = np.argsort(class_1_probabilities)[::-1]
# Select the top 30 indices and extract corresponding node pairs
top_thirty_indices = sorted_indices[:300]
top_thirty_node_pairs = final_2025.iloc[top_thirty_indices]

# Evaluate precision, recall, and accuracy on actual data for these node pairs
predictions_for_top_thirty = classifier.predict(final_2025[columns].iloc[top_thirty_indices])

# Get the first 15 of them:
top_five_node_pairs = final_2025.iloc[top_thirty_indices[:300]]
top_five_probabilities = class_1_probabilities[top_thirty_indices[:300]]

# Print the top 5 node pairs
print("Top 250 Node Pairs:")
for i, (index, row) in enumerate(top_five_node_pairs.iterrows()):
    print(f"Node Pair {i + 1}:")
    print("Prediction Probability:", top_five_probabilities[i])
    print(row)  # Assuming each row represents a node pair
    print()


# Look at the node pairs in Neo4J
query = """
MATCH path = shortestPath((n1)-[*]-(n2))
WHERE id(n1) =   18822 AND id(n2) =    6189
RETURN path
"""







