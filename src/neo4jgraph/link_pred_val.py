import joblib
import pandas as pd
from neo4j import GraphDatabase
import scipy.stats as stats
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Validate the models with the accuracy values of all models
acc_randomforest = [0.87097, 0.85262, 0.93204, 0.99950, 0.90576]
acc_logisticregression = [0.92795, 0.72437, 0.93485, 0.99786, 0.99954]
acc_xgboost = [0.93794, 0.89433, 0.96733, 0.99936, 0.91263]


def calculate_confidence_interval(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    std_error = std_dev / np.sqrt(len(data))
    z_score = stats.norm.ppf(0.975)  # 95% confidence level
    lower_bound = mean - z_score * std_error
    upper_bound = mean + z_score * std_error
    return "{:.6f}".format(mean), "{:.6f}".format(std_error), ("{:.6f}".format(lower_bound), "{:.6f}".format(upper_bound))

mean_rf, std_error_rf, ci_rf = calculate_confidence_interval(acc_randomforest)
mean_lr, std_error_lr, ci_lr = calculate_confidence_interval(acc_logisticregression)
mean_xgb, std_error_xgb, ci_xgb = calculate_confidence_interval(acc_xgboost)

# Print results
print("Random Forest:")
print("  Mean:", mean_rf)
print("  Standard Error:", std_error_rf)
print("  95% Confidence Interval:", ci_rf)
print()
print("Logistic Regression:")
print("  Mean:", mean_lr)
print("  Standard Error:", std_error_lr)
print("  95% Confidence Interval:", ci_lr)
print()
print("XGBoost:")
print("  Mean:", mean_xgb)
print("  Standard Error:", std_error_xgb)
print("  95% Confidence Interval:", ci_xgb)


# XGBoost seems to be the best classifier
# Slot 4 of XGBoost has the best results, so we take model 4 of logreg for evaluation.

# calculate the positive and negative links WITHOUT rebalancing per class weights for validation
def pos_neg_test_val():
    uri = "bolt://localhost:7002"  # Update with your Neo4j connection URI
    driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))

    with driver.session(database="neo4j") as session:
        result_positive = session.run(f"""
            MATCH (second_website)-[:SLOT_VAL_TEST]->(stakeholder_pers)
            RETURN id(second_website) AS node1, id(stakeholder_pers) AS node2, 1 AS label
            UNION
            MATCH (second_website)-[:SLOT_VAL_TEST]->(stakeholder_org)
            RETURN id(second_website) AS node1, id(stakeholder_org) AS node2, 1 AS label
            UNION
            MATCH (stakeholder_org)-[:SLOT_VAL_TEST]->(stakeholder_pers)
            RETURN id(stakeholder_org) AS node1, id(stakeholder_pers) AS node2, 1 AS label
            """)
        # Build a dataframe from positive links
        test_existing_links = pd.DataFrame([dict(record) for record in result_positive])

        result_negative = session.run(f"""
            MATCH (second_website)-[:SLOT_VAL_TEST*2]-(stakeholder_pers)
            WHERE NOT((second_website)-[:SLOT_VAL_TEST]-(stakeholder_pers))
            RETURN id(second_website) AS node1, id(stakeholder_pers) AS node2, 0 AS label
            UNION
            MATCH (second_website)-[:SLOT_VAL_TEST*2]-(stakeholder_org)
            WHERE NOT((second_website)-[:SLOT_VAL_TEST]-(stakeholder_org))
            RETURN id(second_website) AS node1, id(stakeholder_org) AS node2, 0 AS label
            UNION
            MATCH (stakeholder_org)-[:SLOT_VAL_TEST*2]-(stakeholder_pers)
            WHERE NOT((stakeholder_org)-[:SLOT_VAL_TEST]-(stakeholder_pers))
            RETURN id(stakeholder_org) AS node1, id(stakeholder_pers) AS node2, 0 AS label
            """)
        # Build a dataframe from negative links
        test_missing_links = pd.DataFrame([dict(record) for record in result_negative])
        test_missing_links = test_missing_links.drop_duplicates()
        test_df = pd.concat([test_missing_links, test_existing_links], ignore_index=True)
        test_df['label'] = test_df['label'].astype('category')

        # Concatenate the datasets and categorize them via label 1 or 2
        count_class_0, count_class_1 = test_df.label.value_counts()
        print(f"Negative examples of testing slot validation: {count_class_0}")
        print(f"Positive examples of testing slot validation: {count_class_1}")

        df_class_0 = test_df[test_df['label'] == 0]
        df_class_1 = test_df[test_df['label'] == 1]

        df_split_test = pd.concat([df_class_0, df_class_1], axis=0)
        df_split_test.to_csv(f'df_split_val_test.csv', index=False)


pos_neg_test_val()

# feature engineering
# link prediction features: common neighbors score, preferential attachment score, adamic adar score,
# total neighbors score and resource allocation score

def apply_graphy_features(data, rel_type):
    uri = "bolt://localhost:7002"  # Update with your Neo4j connection URI
    driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))

    # Calculate the feature values of each year
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
    pairs = [{"node1": node1, "node2": node2} for node1, node2 in data[["node1", "node2"]].values.tolist()]

    # Query the feature values per node pair and store in dataframe
    with driver.session(database="neo4j") as session:
        result = session.run(query, {"pairs": pairs, "relType": rel_type})
        features = pd.DataFrame([dict(record) for record in result])
    return pd.merge(data, features, on = ["node1", "node2"])


# Execute the feature calculations
df_split_test = pd.read_csv(f'df_split_val_test.csv')
df_split_test = apply_graphy_features(df_split_test, f"SLOT_VAL_TEST")

print(df_split_test.sample(5, random_state=42))
df_split_test.to_csv(f'val_test.csv', index=False)
df_split_test = pd.read_csv(f'val_test.csv')

# Evaluate the models prediction accuracy, precision and recall
def evaluate_model(predictions, actual):
    actual = np.array(actual)
    return pd.DataFrame({
        "Measure": ["Accuracy", "Precision", "Recall"],
        "Score": [accuracy_score(actual, predictions),
                  precision_score(actual, predictions),
                  recall_score(actual, predictions)]
    })


# load best model: XGBoost model slot 4
classifier = joblib.load('model_slot_4.pkl')

columns = [
    "cn", "pa", "aa", "tn", "ra"
]

# The predictions are computed using the before calculated classifier on the feature set of the test split
predictions = classifier.predict(df_split_test[columns])
y_test = df_split_test["label"]
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
predictions = classifier.predict(df_split_test[columns])
y_test = df_split_test["label"]
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
predicted_probabilities = classifier.predict_proba(df_split_test[columns])
# Extract probabilities for class 1
class_1_probabilities = predicted_probabilities[:, 1]
# Sort indices based on probabilities in descending order
sorted_indices = np.argsort(class_1_probabilities)[::-1]
# Select the top 30 indices and extract corresponding node pairs
top_thirty_indices = sorted_indices[:30]
top_thirty_node_pairs = df_split_test.iloc[top_thirty_indices]

# Evaluate precision, recall, and accuracy on actual data for these node pairs
actual_data = df_split_test["label"].iloc[top_thirty_indices]
predictions_for_top_thirty = classifier.predict(df_split_test[columns].iloc[top_thirty_indices])

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
top_five_node_pairs = df_split_test.iloc[top_thirty_indices[:5]]
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
probs = classifier.predict_proba(df_split_test[columns])[:, 1]

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



