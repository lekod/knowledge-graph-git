# link prediction pipeline as a combination from Neo4J
# (https://neo4j.com/docs/graph-data-science/current/machine-learning/linkprediction-pipelines/config/)
# and https://neo4j.com/developer/graph-data-science/link-prediction/scikit-learn/
# and Philipp Brunenberg (https://www.youtube.com/watch?v=kq_b0QmxFCI)

import pandas as pd
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
from IPython.display import display
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (recall_score, precision_score, accuracy_score, roc_curve, auc, precision_recall_curve,
                             classification_report, confusion_matrix)
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import numpy as np
import joblib
from tabulate import tabulate
import yaml
from sklearn.utils.class_weight import compute_class_weight
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

# Use the SLOT_NUM variable from the config file to compute slot 1-5, validation and prediction slot
slot_num = config['SLOT_NUM']

# Links in the knowledge graph receive the data from the nodes
def Links_get_date():
    uri = "bolt://localhost:7002"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))

    # The date of the link is the date of the node that came in latest
    links_date_query = """
        MATCH (n)-[r:LINKS_TO]->(m)
        WHERE m.date_up IS NOT NULL AND n.date_up IS NOT NULL
        WITH n, r, m, 
        CASE 
        WHEN n.date_up > m.date_up THEN n.date_up
        ELSE m.date_up
        END AS latest_date
        SET r.date_up = latest_date
        RETURN n, r, m
        """

    with driver.session(database="neo4j") as session:
        session.run(links_date_query)


# The years, when the links were created are plottet
def plot_years():
    uri = "bolt://localhost:7002"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))

    plt.style.use('fivethirtyeight')
    pd.set_option('display.float_format', lambda x: '%.3f' % x)

    # The links are set first
    set_year_query = """
    MATCH (n)-[r:LINKS_TO]->(m)
    WHERE r.date_up IS NOT NULL AND n.date_up IS NOT NULL AND m.date_up IS NOT NULL 
    WITH r, SUBSTRING(r.date_up, 0, 4) AS year
    SET r.year = toString(year)
    """

    with driver.session(database="neo4j") as session:
        session.run(set_year_query)

    # Then the links years are counted
    count_years_query = """
    MATCH (n)-[r:LINKS_TO]->(m)
    WHERE r.date_up IS NOT NULL AND n.date_up IS NOT NULL AND m.date_up IS NOT NULL AND toInteger(r.year) >= 2008
    WITH r.year AS year, count(*) AS count
    ORDER BY year
    RETURN year, count
    """

    with driver.session(database="neo4j") as session:
        result = session.run(count_years_query)
        by_year = pd.DataFrame([dict(record) for record in result])

    # plot the year plot
    ax = by_year.plot(kind='bar', x='year', y='count', legend=None, figsize=(12,7), color='#1874CD')
    ax.yaxis.set_label_text("Number of links added")
    ax.xaxis.set_label_text("")
    plt.tight_layout()
    plt.style.use('ggplot')
    plt.show()

plot_years()


# Split in 5 training and test sets, one validation split and one prediction split
def split_train_test():
    uri = "bolt://localhost:7002"  # Update with your Neo4j connection URI
    driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))

    # build the 5-fold rolling split:
    # SPLIT 1
    query = """
        MATCH (a)-[r:LINKS_TO]->(b)
        WHERE r.date_up IS NOT NULL AND a.date_up IS NOT NULL AND b.date_up IS NOT NULL AND toInteger(r.year) >= 2009 AND toInteger(r.year) <= 2018
        MERGE (a)-[:SLOT_1_TRAIN {year: r.year}]-(b);
        """

    with driver.session(database="neo4j") as session:
        display(session.run(query).consume().counters)


    # SPLIT 1 TEST
    query = """
        MATCH (a)-[r:LINKS_TO]->(b)
        WHERE r.date_up IS NOT NULL AND a.date_up IS NOT NULL AND b.date_up IS NOT NULL AND toInteger(r.year) = 2019
        MERGE (a)-[:SLOT_1_TEST {year: r.year}]-(b);
        """

    with driver.session(database="neo4j") as session:
        display(session.run(query).consume().counters)

    # SPLIT 2
    query = """
            MATCH (a)-[r:LINKS_TO]->(b)
            WHERE r.date_up IS NOT NULL AND a.date_up IS NOT NULL AND b.date_up IS NOT NULL AND toInteger(r.year) >= 2009 AND toInteger(r.year) <= 2019
            MERGE (a)-[:SLOT_2_TRAIN {year: r.year}]-(b);
            """

    with driver.session(database="neo4j") as session:
        display(session.run(query).consume().counters)

    # SPLIT 2 TEST
    query = """
            MATCH (a)-[r:LINKS_TO]->(b)
            WHERE r.date_up IS NOT NULL AND a.date_up IS NOT NULL AND b.date_up IS NOT NULL AND toInteger(r.year) = 2020
            MERGE (a)-[:SLOT_2_TEST {year: r.year}]-(b);
            """

    with driver.session(database="neo4j") as session:
        display(session.run(query).consume().counters)

    # SPLIT 3
    query = """
            MATCH (a)-[r:LINKS_TO]->(b)
            WHERE r.date_up IS NOT NULL AND a.date_up IS NOT NULL AND b.date_up IS NOT NULL AND toInteger(r.year) >= 2009 AND toInteger(r.year) <= 2020
            MERGE (a)-[:SLOT_3_TRAIN {year: r.year}]-(b);
            """

    with driver.session(database="neo4j") as session:
        display(session.run(query).consume().counters)

    # SPLIT 3 TEST
    query = """
            MATCH (a)-[r:LINKS_TO]->(b)
            WHERE r.date_up IS NOT NULL AND a.date_up IS NOT NULL AND b.date_up IS NOT NULL AND toInteger(r.year) = 2021
            MERGE (a)-[:SLOT_3_TEST {year: r.year}]-(b);
            """

    with driver.session(database="neo4j") as session:
        display(session.run(query).consume().counters)

    # SPLIT 4
    query = """
               MATCH (a)-[r:LINKS_TO]->(b)
               WHERE r.date_up IS NOT NULL AND a.date_up IS NOT NULL AND b.date_up IS NOT NULL AND toInteger(r.year) >= 2009 AND toInteger(r.year) <= 2021
               MERGE (a)-[:SLOT_4_TRAIN {year: r.year}]-(b);
               """

    with driver.session(database="neo4j") as session:
        display(session.run(query).consume().counters)

    # SPLIT 4 TEST
    query = """
               MATCH (a)-[r:LINKS_TO]->(b)
               WHERE r.date_up IS NOT NULL AND a.date_up IS NOT NULL AND b.date_up IS NOT NULL AND toInteger(r.year) = 2022
               MERGE (a)-[:SLOT_4_TEST {year: r.year}]-(b);
               """

    with driver.session(database="neo4j") as session:
        display(session.run(query).consume().counters)


    # SPLIT 5
    query = """
               MATCH (a)-[r:LINKS_TO]->(b)
               WHERE r.date_up IS NOT NULL AND a.date_up IS NOT NULL AND b.date_up IS NOT NULL AND toInteger(r.year) >= 2009 AND toInteger(r.year) <= 2022
               MERGE (a)-[:SLOT_5_TRAIN {year: r.year}]-(b);
               """

    with driver.session(database="neo4j") as session:
        display(session.run(query).consume().counters)

    # SPLIT 5 TEST
    query = """
               MATCH (a)-[r:LINKS_TO]->(b)
               WHERE r.date_up IS NOT NULL AND a.date_up IS NOT NULL AND b.date_up IS NOT NULL AND toInteger(r.year) = 2023
               MERGE (a)-[:SLOT_5_TEST {year: r.year}]-(b);
               """

    with driver.session(database="neo4j") as session:
        display(session.run(query).consume().counters)


    # Validation split
    query = """
               MATCH (a)-[r:LINKS_TO]->(b)
               WHERE r.date_up IS NOT NULL AND a.date_up IS NOT NULL AND b.date_up IS NOT NULL AND toInteger(r.year) >= 2009 AND toInteger(r.year) <= 2023
               MERGE (a)-[:SLOT_VAL_TRAIN {year: r.year}]-(b);
               """

    with driver.session(database="neo4j") as session:
        display(session.run(query).consume().counters)

    query = """
               MATCH (a)-[r:LINKS_TO]->(b)
               WHERE r.date_up IS NOT NULL AND a.date_up IS NOT NULL AND b.date_up IS NOT NULL AND toInteger(r.year) = 2024
               MERGE (a)-[:SLOT_VAL_TEST {year: r.year}]-(b);
               """

    with driver.session(database="neo4j") as session:
        display(session.run(query).consume().counters)

# Prediction split

    query = """
               MATCH (a)-[r:LINKS_TO]->(b)
               WHERE r.date_up IS NOT NULL AND a.date_up IS NOT NULL AND b.date_up IS NOT NULL AND toInteger(r.year) >= 2009 AND toInteger(r.year) <= 2024
               MERGE (a)-[:SLOT_PRED_TEST {year: r.year}]-(b);
               """

    with driver.session(database="neo4j") as session:
        display(session.run(query).consume().counters)

# Execute the split building
split_train_test()


# Make a summary of how many train and test data is used in the splits
def summarize_splits():

    # Connect to Neo4J
    uri = "bolt://localhost:7002"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))

    split_counts = {}

    # Define the splits and their corresponding Cypher queries
    splits = {
        1: ("SLOT_1_TRAIN", "SLOT_1_TEST"),
        2: ("SLOT_2_TRAIN", "SLOT_2_TEST"),
        3: ("SLOT_3_TRAIN", "SLOT_3_TEST"),
        4: ("SLOT_4_TRAIN", "SLOT_4_TEST"),
        5: ("SLOT_5_TRAIN", "SLOT_5_TEST"),
        "validation": ("SLOT_VAL_TRAIN", "SLOT_VAL_TEST")
    }

    # Execute cypher queries and store the counts in the split_counts dictionary
    for split, (train_rel, test_rel) in splits.items():
        train_query = f"MATCH ()-[:{train_rel}]->() RETURN COUNT(*) AS train_count"
        test_query = f"MATCH ()-[:{test_rel}]->() RETURN COUNT(*) AS test_count"

        with driver.session(database="neo4j") as session:
            train_count = session.run(train_query).single()["train_count"]
            test_count = session.run(test_query).single()["test_count"]

            total_count = train_count + test_count
            train_percentage = (train_count / total_count) * 100
            test_percentage = (test_count / total_count) * 100

            split_counts[split] = {"train_count": train_count, "train_percentage": train_percentage,
                                   "test_count": test_count, "test_percentage": test_percentage}

    # Define the headers of the table
    headers = ["Split", "Train Count", "Train Percentage", "Test Count", "Test Percentage"]
    rows = []

    # Compute the table
    for split, counts in split_counts.items():
        row = [split, counts["train_count"], f"{counts['train_percentage']:.2f}%", counts["test_count"],
               f"{counts['test_percentage']:.2f}%"]
        rows.append(row)

    print(tabulate(rows, headers=headers))

summarize_splits()

# Build a dataset of positive and negative links in the network
def pos_neg_train(slot_num):

    # Connect to Neo4j
    uri = "bolt://localhost:7002"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))

    # Calculate positive results: Every node pairs connected in the train split from website to stakeholder Person or
    # to stakeholder organisation or stakeholder to stakeholder
    with driver.session(database="neo4j") as session:

        result_positive = session.run(f"""
            MATCH (second_website)-[rel:SLOT_{slot_num}_TRAIN]->(stakeholder_pers)
            RETURN id(second_website) AS node1, id(stakeholder_pers) AS node2, 1 AS label
            UNION
            MATCH (second_website)-[rel:SLOT_{slot_num}_TRAIN]->(stakeholder_org)
            RETURN id(second_website) AS node1, id(stakeholder_org) AS node2, 1 AS label
            UNION
            MATCH (stakeholder_org)-[rel:SLOT_{slot_num}_TRAIN]->(stakeholder_pers)
            RETURN id(stakeholder_org) AS node1, id(stakeholder_pers) AS node2, 1 AS label
        """)
        # Build a dataframe from positive links
        train_existing_links = pd.DataFrame([dict(record) for record in result_positive])

        # Negative examples: Node pairs not directly connected by the SLOT_{slot_num}_TRAIN relationship
        # but 1-2 steps away
        result_negative = session.run(f"""
                    MATCH (second_website)-[:SLOT_{slot_num}_TRAIN*2]-(stakeholder_pers)
                    WHERE NOT((second_website)-[:SLOT_{slot_num}_TRAIN]-(stakeholder_pers))
                    RETURN id(second_website) AS node1, id(stakeholder_pers) AS node2, 0 AS label
                    UNION
                    MATCH (second_website)-[:SLOT_{slot_num}_TRAIN*2]-(stakeholder_org)
                    WHERE NOT((second_website)-[:SLOT_{slot_num}_TRAIN]-(stakeholder_org))
                    RETURN id(second_website) AS node1, id(stakeholder_org) AS node2, 0 AS label
                    UNION
                    MATCH (stakeholder_org)-[:SLOT_{slot_num}_TRAIN*2]-(stakeholder_pers)
                    WHERE NOT((stakeholder_org)-[:SLOT_{slot_num}_TRAIN]-(stakeholder_pers))
                    RETURN id(stakeholder_org) AS node1, id(stakeholder_pers) AS node2, 0 AS label
                """)

        # Build a dataframe from negative links
        train_missing_links = pd.DataFrame([dict(record) for record in result_negative])
        train_missing_links = train_missing_links.drop_duplicates()

        # Concatenate the datasets and categorize them via label 1 or 2
        training_df = pd.concat([train_missing_links, train_existing_links], ignore_index=True)
        training_df['label'] = training_df['label'].astype('category')

        count_class_0, count_class_1 = training_df.label.value_counts()
        print(f"Negative examples train slot {slot_num}: {count_class_0}")
        print(f"Positive examples train slot {slot_num}: {count_class_1}")

        df_class_0 = training_df[training_df['label'] == 0]
        df_class_1 = training_df[training_df['label'] == 1]

        # Save to split to a csv
        df_split_train = pd.concat([df_class_0, df_class_1], axis=0)
        df_split_train.to_csv(f'df_split_{slot_num}_train.csv', index=False)

        # Print the distribution of positive and negative links
        print(f'Class distribution of train slot {slot_num}:')
        print(df_split_train.label.value_counts())
        print(df_split_train.sample(5, random_state=42))

        # Compute class weights since we have imbalanced data
        class_labels = np.array(df_split_train['label'])
        class_weights = compute_class_weight('balanced', classes=np.unique(class_labels), y=class_labels)
        class_weight_dict = dict(zip(np.unique(class_labels), class_weights))
        print("Class Weights:", class_weight_dict)

        # Compute just the class weights for the randomforest or logistic regression mod
        num_negative_samples = np.sum(class_labels == 0)
        num_positive_samples = np.sum(class_labels == 1)
        scale_pos_weight = num_negative_samples / num_positive_samples

        return scale_pos_weight

        # Compute just the class weights for the randomforest or logistic regression model
        # return class_weight_dict

pos_neg_train(slot_num)

# Also compute the positive and negative links for the test slots
def pos_neg_test(slot_num):

    # Connect to Neo4j
    uri = "bolt://localhost:7002"  # Update with your Neo4j connection URI
    driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))

    # Calculate positive results: Every node pairs connected in the test split from website to stakeholder Person or
    # to stakeholder organisation or stakeholder to stakeholder
    with driver.session(database="neo4j") as session:
        result_positive = session.run(f"""
                    MATCH (second_website)-[:SLOT_{slot_num}_TEST]->(stakeholder_pers)
                    RETURN id(second_website) AS node1, id(stakeholder_pers) AS node2, 1 AS label
                    UNION
                    MATCH (second_website)-[:SLOT_{slot_num}_TEST]->(stakeholder_org)
                    RETURN id(second_website) AS node1, id(stakeholder_org) AS node2, 1 AS label
                    UNION
                    MATCH (stakeholder_org)-[:SLOT_{slot_num}_TEST]->(stakeholder_pers)
                    RETURN id(stakeholder_org) AS node1, id(stakeholder_pers) AS node2, 1 AS label
            """)
        # Build a dataframe from positive links
        test_existing_links = pd.DataFrame([dict(record) for record in result_positive])

        # Negative examples: Node pairs not directly connected by the SLOT_{slot_num}_TRAIN relationship
        # but 1-2 steps away

        result_negative = session.run(f"""
                    MATCH (second_website)-[:SLOT_{slot_num}_TEST*2]-(stakeholder_pers)
                    WHERE NOT((second_website)-[:SLOT_{slot_num}_TEST]-(stakeholder_pers))
                    RETURN id(second_website) AS node1, id(stakeholder_pers) AS node2, 0 AS label
                    UNION
                    MATCH (second_website)-[:SLOT_{slot_num}_TEST*2]-(stakeholder_org)
                    WHERE NOT((second_website)-[:SLOT_{slot_num}_TEST]-(stakeholder_org))
                    RETURN id(second_website) AS node1, id(stakeholder_org) AS node2, 0 AS label
                    UNION
                    MATCH (stakeholder_org)-[:SLOT_{slot_num}_TEST*2]-(stakeholder_pers)
                    WHERE NOT((stakeholder_org)-[:SLOT_{slot_num}_TEST]-(stakeholder_pers))
                    RETURN id(stakeholder_org) AS node1, id(stakeholder_pers) AS node2, 0 AS label
                """)

        # Build a dataframe from negative links
        test_missing_links = pd.DataFrame([dict(record) for record in result_negative])
        test_missing_links = test_missing_links.drop_duplicates()
        test_df = pd.concat([test_missing_links, test_existing_links], ignore_index=True)
        test_df['label'] = test_df['label'].astype('category')

        # Concatenate the datasets and categorize them via label 1 or 2
        count_class_0, count_class_1 = test_df.label.value_counts()
        print(f"Negative examples of testing slot {slot_num}: {count_class_0}")
        print(f"Positive examples of testing slot {slot_num}: {count_class_1}")

        df_class_0 = test_df[test_df['label'] == 0]
        df_class_1 = test_df[test_df['label'] == 1]

        df_split_test = pd.concat([df_class_0, df_class_1], axis=0)
        df_split_test.to_csv(f'df_split_{slot_num}_test.csv', index=False)

        # Print the class distribution in the test slot
        print(f'Class distribution of test slot {slot_num}:')
        print(df_split_test.label.value_counts())
        print(df_split_test.sample(5, random_state=42))

pos_neg_test(slot_num)

# feature engineering
# link prediction features: common neighbors score, preferential attachment score, adamic adar score,
# total neighbors score and resource allocation score

def apply_graphy_features(data, rel_type):
    # Connect to Neo4j database
    uri = "bolt://localhost:7002"
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
    # Calculate per node pair
    pairs = [{"node1": node1, "node2": node2} for node1, node2 in data[["node1", "node2"]].values.tolist()]

    # Query the feature values per node pair and store in dataframe
    with driver.session(database="neo4j") as session:
        result = session.run(query, {"pairs": pairs, "relType": rel_type})
        features = pd.DataFrame([dict(record) for record in result])
    return pd.merge(data, features, on = ["node1", "node2"])

# Execute the feature calculations
df_split_train = pd.read_csv(f'df_split_{slot_num}_train.csv')
df_split_test = pd.read_csv(f'df_split_{slot_num}_test.csv')
df_split_train = apply_graphy_features(df_split_train, f"SLOT_{slot_num}_TRAIN")
df_split_test = apply_graphy_features(df_split_test, f"SLOT_{slot_num}_TEST")

# Print the sample and store the split in a csv
print(df_split_train.sample(5, random_state=42))
print(df_split_test.sample(5, random_state=42))
df_split_train.to_csv(f'df_split_{slot_num}_train2.csv', index=False)
df_split_test.to_csv(f'df_split_{slot_num}_test2.csv', index=False)

# Load the test CSV file into a DataFrame
df_split_train = pd.read_csv(f'df_split_{slot_num}_train2.csv')
df_split_test = pd.read_csv(f'df_split_{slot_num}_test2.csv')

# Train the model with different classifiers like randomforest, logisticregression and xgboost
# For xgboost we use the scale_pos_weight:
def train_model(df_split_train, scale_pos_weight):
# For randomforest and logistic regression we use the class weights:
# def train_model(df_split_train, class_weights):

    # Choose the model you want to execute:
    # classifier = RandomForestClassifier(n_estimators=30, max_depth=10, random_state=0, class_weight=class_weights)
    # classifier = LogisticRegression(random_state=0, class_weight=class_weights)
    classifier = xgb.XGBClassifier(objective="binary:logistic", random_state=0, scale_pos_weight = scale_pos_weight)

    # Define the columns that are used as prediction
    columns = ["cn", "pa", "aa", "tn", "ra"]

    # Define the X and Y value that are used in the classifier
    X = df_split_train[columns]
    y = df_split_train["label"]

    classifier.fit(X, y)
    return classifier, columns

# For xgboost we use the scale_pos_weight:
scale_pos_weight = pos_neg_train(slot_num)

# For randomforest and logistic regression we use the class weights:
# classifier, columns = train_model(df_split_train, class_weights= pos_neg_train(slot_num))
classifier, columns = train_model(df_split_train, scale_pos_weight=scale_pos_weight)

# Calculate feature importance to understand which features contribute more to the classification
def feature_importance(columns, classifier):
    # Create a tuple list with the features and their importance
    features = list(zip(columns, classifier.feature_importances_))
    # Rank their importance
    sorted_features = sorted(features, key=lambda x: x[1] * -1)

    # Extract feature and importance names
    keys = [value[0] for value in sorted_features]
    values = [value[1] for value in sorted_features]
    # print the table
    print(pd.DataFrame(data={'feature': keys, 'value': values}))

print("-" * 55)
# feature_importance(columns, classifier)

# Evaluate the models prediction accuracy, precision and recall
def evaluate_model(predictions, actual):
    actual = np.array(actual)
    return pd.DataFrame({
        "Measure": ["Accuracy", "Precision", "Recall"],
        "Score": [accuracy_score(actual, predictions),
                  precision_score(actual, predictions),
                  recall_score(actual, predictions)]
    })

# The predictions are computed using the before calculated classifier on the feature set of the test split
predictions = classifier.predict(df_split_test[columns])

# The actual label are the labels in the test set
actual = df_split_test["label"]

print("-" * 55)
print(f'Evaluation split {slot_num}:')
print(evaluate_model(predictions, actual))

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
        print(f"Metrics for {class_label} in split {slot_num}:")
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
plt.title(f'Area Under Curve (AUC) for slot {slot_num}')
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
plt.title(f'Area Under Precision-Recall Curve (AUPRC) for slot {slot_num}')
plt.legend(loc="lower left")
plt.show()
print("AUPRC:", auprc)

# Store the model
joblib.dump(classifier, f'model_slot_{slot_num}.pkl')