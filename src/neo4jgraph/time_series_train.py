import pandas as pd
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
from IPython.display import display
import numpy as np
import sqlite3
from keras.optimizers import legacy
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.layers import LSTM, GRU, Dropout, Dense, Bidirectional
import yaml
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

slotnum = config['SLOTNUM_TIME']
slot_year = config['SLOTYEAR_TIME']


def Links_get_date():
    uri = "bolt://localhost:7002"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))

    # The date of the link is the date of the node that came in latest
    set_link_query = """
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
        session.run(set_link_query)

# Plot the links that are added throughout the years
def plot_years():
    # Connect to Neo4j
    uri = "bolt://localhost:7002"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))

    # Configure plot format
    plt.style.use('fivethirtyeight')
    pd.set_option('display.float_format', lambda x: '%.3f' % x)

    # Run queries to get just the year, not the rest of the date
    set_year_query = """
    MATCH (n)-[r:LINKS_TO]->(m)
    WHERE r.date_up IS NOT NULL AND n.date_up IS NOT NULL AND m.date_up IS NOT NULL 
    WITH r, SUBSTRING(r.date_up, 0, 4) AS year
    SET r.year = toString(year)
    """
    with driver.session(database="neo4j") as session:
        session.run(set_year_query)

    # Run query to count the year
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

    # plot it
    ax = by_year.plot(kind='bar', x='year', y='count', legend=None, figsize=(12,7), color='#1874CD')
    ax.yaxis.set_label_text("Number of links added")
    ax.xaxis.set_label_text("")
    plt.tight_layout()
    plt.style.use('ggplot')
    plt.show()

plot_years()

# Create yearly splits to represent the network of a certain year
def create_year_splits():
    uri = "bolt://localhost:7002"
    driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))

    # SPLIT 2010:

    query = """
            MATCH (a)-[r:LINKS_TO]->(b)
            WHERE r.date_up IS NOT NULL AND a.date_up IS NOT NULL AND b.date_up IS NOT NULL AND toInteger(r.year) = 2010
            MERGE (a)-[:SLOT_2010 {year: r.year}]-(b);
            """
    with driver.session(database="neo4j") as session:
        display(session.run(query).consume().counters)

    # SPLIT 2011:
    query = """
            MATCH (a)-[r:LINKS_TO]->(b)
            WHERE r.date_up IS NOT NULL AND a.date_up IS NOT NULL AND b.date_up IS NOT NULL AND toInteger(r.year) >= 2010 AND toInteger(r.year) <= 2011
            MERGE (a)-[:SLOT_2011 {year: r.year}]-(b);
            """
    with driver.session(database="neo4j") as session:
        display(session.run(query).consume().counters)

    # SPLIT 2012
    query = """
            MATCH (a)-[r:LINKS_TO]->(b)
            WHERE r.date_up IS NOT NULL AND a.date_up IS NOT NULL AND b.date_up IS NOT NULL AND toInteger(r.year) >= 2010 AND toInteger(r.year) <= 2012
            MERGE (a)-[:SLOT_2012 {year: r.year}]-(b);
            """

    with driver.session(database="neo4j") as session:
        display(session.run(query).consume().counters)


    # SPLIT 2013
    query = """
            MATCH (a)-[r:LINKS_TO]->(b)
            WHERE r.date_up IS NOT NULL AND a.date_up IS NOT NULL AND b.date_up IS NOT NULL AND toInteger(r.year) >= 2010 AND toInteger(r.year) <= 2013
            MERGE (a)-[:SLOT_2013 {year: r.year}]-(b);
            """

    with driver.session(database="neo4j") as session:
        display(session.run(query).consume().counters)


    # SPLIT 2014
    query = """
            MATCH (a)-[r:LINKS_TO]->(b)
            WHERE r.date_up IS NOT NULL AND a.date_up IS NOT NULL AND b.date_up IS NOT NULL AND toInteger(r.year) >= 2010 AND toInteger(r.year) <= 2014
            MERGE (a)-[:SLOT_2014 {year: r.year}]-(b);
            """

    with driver.session(database="neo4j") as session:
        display(session.run(query).consume().counters)

    # SPLIT 2015

    query = """
            MATCH (a)-[r:LINKS_TO]->(b)
            WHERE r.date_up IS NOT NULL AND a.date_up IS NOT NULL AND b.date_up IS NOT NULL AND toInteger(r.year) >= 2010 AND toInteger(r.year) <= 2015
            MERGE (a)-[:SLOT_2015 {year: r.year}]-(b);
            """

    with driver.session(database="neo4j") as session:
        display(session.run(query).consume().counters)

    # SPLIT 2016

    query = """
            MATCH (a)-[r:LINKS_TO]->(b)
            WHERE r.date_up IS NOT NULL AND a.date_up IS NOT NULL AND b.date_up IS NOT NULL AND toInteger(r.year) >= 2010 AND toInteger(r.year) <= 2016
            MERGE (a)-[:SLOT_2016 {year: r.year}]-(b);
            """

    with driver.session(database="neo4j") as session:
        display(session.run(query).consume().counters)

    # SPLIT 2017

    query = """
            MATCH (a)-[r:LINKS_TO]->(b)
            WHERE r.date_up IS NOT NULL AND a.date_up IS NOT NULL AND b.date_up IS NOT NULL AND toInteger(r.year) >= 2010 AND toInteger(r.year) <= 2017
            MERGE (a)-[:SLOT_2017 {year: r.year}]-(b);
            """

    with driver.session(database="neo4j") as session:
        display(session.run(query).consume().counters)

    # SPLIT 2018

    query = """
           MATCH (a)-[r:LINKS_TO]->(b)
           WHERE r.date_up IS NOT NULL AND a.date_up IS NOT NULL AND b.date_up IS NOT NULL AND toInteger(r.year) >= 2010 AND toInteger(r.year) <= 2018
           MERGE (a)-[:SLOT_2018 {year: r.year}]-(b);
           """

    with driver.session(database="neo4j") as session:
        display(session.run(query).consume().counters)

    # SPLIT 2019

    query = """
           MATCH (a)-[r:LINKS_TO]->(b)
           WHERE r.date_up IS NOT NULL AND a.date_up IS NOT NULL AND b.date_up IS NOT NULL AND toInteger(r.year) >= 2010 AND toInteger(r.year) <= 2019
           MERGE (a)-[:SLOT_2019 {year: r.year}]-(b);
           """

    with driver.session(database="neo4j") as session:
        display(session.run(query).consume().counters)


    # SPLIT 2020

    query = """
           MATCH (a)-[r:LINKS_TO]->(b)
           WHERE r.date_up IS NOT NULL AND a.date_up IS NOT NULL AND b.date_up IS NOT NULL AND toInteger(r.year) >= 2010 AND toInteger(r.year) <= 2020
           MERGE (a)-[:SLOT_2020 {year: r.year}]-(b);
           """

    with driver.session(database="neo4j") as session:
        display(session.run(query).consume().counters)

    # SPLIT 2021

    query = """
           MATCH (a)-[r:LINKS_TO]->(b)
           WHERE r.date_up IS NOT NULL AND a.date_up IS NOT NULL AND b.date_up IS NOT NULL AND toInteger(r.year) >= 2010 AND toInteger(r.year) <= 2021
           MERGE (a)-[:SLOT_2021 {year: r.year}]-(b);
           """

    with driver.session(database="neo4j") as session:
        display(session.run(query).consume().counters)

    # SPLIT 2022

    query = """
           MATCH (a)-[r:LINKS_TO]->(b)
           WHERE r.date_up IS NOT NULL AND a.date_up IS NOT NULL AND b.date_up IS NOT NULL AND toInteger(r.year) >= 2010 AND toInteger(r.year) <= 2022
           MERGE (a)-[:SLOT_2022 {year: r.year}]-(b);
           """

    with driver.session(database="neo4j") as session:
        display(session.run(query).consume().counters)

    # SPLIT 2023

    query = """
          MATCH (a)-[r:LINKS_TO]->(b)
          WHERE r.date_up IS NOT NULL AND a.date_up IS NOT NULL AND b.date_up IS NOT NULL AND toInteger(r.year) >= 2010 AND toInteger(r.year) <= 2023
          MERGE (a)-[:SLOT_2023 {year: r.year}]-(b);
          """

    with driver.session(database="neo4j") as session:
        display(session.run(query).consume().counters)


    # Validation split 2024

    query = """
           MATCH (a)-[r:LINKS_TO]->(b)
           WHERE r.date_up IS NOT NULL AND a.date_up IS NOT NULL AND b.date_up IS NOT NULL AND toInteger(r.year) >= 2010 AND toInteger(r.year) <= 2024
           MERGE (a)-[:SLOT_2024 {year: r.year}]-(b);
           """

    with driver.session(database="neo4j") as session:
        display(session.run(query).consume().counters)

create_year_splits()

# This function filters out all the given links and creates their node pairs
def pos_node_pairs(slot_year):

    # Connect to Neo4j database
    uri = "bolt://localhost:7002"  # Update with your Neo4j connection URI
    driver = GraphDatabase.driver(uri, auth=("neo4j", "password"))

    # Get all links between websites and stakeholders and stakeholders to stakeholders that do not exist yet
    with driver.session(database="neo4j") as session:

        result_negative = session.run(f"""
            MATCH (second_website)-[:SLOT_{slot_year}*2]-(stakeholder_pers)
            WHERE NOT((second_website)-[:SLOT_{slot_year}]-(stakeholder_pers))
            RETURN id(second_website) AS node1, id(stakeholder_pers) AS node2, 0 AS label
            UNION
            MATCH (second_website)-[:SLOT_{slot_year}*2]-(stakeholder_org)
            WHERE NOT((second_website)-[:SLOT_{slot_year}]-(stakeholder_org))
            RETURN id(second_website) AS node1, id(stakeholder_org) AS node2, 0 AS label
            UNION
            MATCH (stakeholder_org)-[:SLOT_{slot_year}*2]-(stakeholder_pers)
            WHERE NOT((stakeholder_org)-[:SLOT_{slot_year}]-(stakeholder_pers))
            RETURN id(stakeholder_org) AS node1, id(stakeholder_pers) AS node2, 0 AS label
        """)
        missing_links = pd.DataFrame([dict(record) for record in result_negative])
        missing_links = missing_links.drop_duplicates()
        missing_links.to_csv(f'df_split_{slot_year}.csv', index=False)

pos_node_pairs(slot_year)

# feature engineering
# link prediction features: common neighbors score, preferential attachment score, total neighbors score,
# adamic adar score, totalneighbors, resource allocation

# Calculate the feature values of each year
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

# Since some datasets are too big and overload the docker cpu, big datasets are chunked and processed one after another
def chunk_dataframe(df, n_chunks):
    chunk_size = len(df) // n_chunks
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    return chunks

# Read node to node csv and insert the year created
df_split_year = pd.read_csv(f'df_split_{slot_year}.csv')
df_split_year.insert(0, "year", slot_year)

if slot_year == 2024:
    n_chunks = 2
    df_chunks = chunk_dataframe(df_split_year, n_chunks)
    df_results = pd.DataFrame()

    # Loop through each chunk, process it, and concatenate the results
    for idx, df_chunk in enumerate(df_chunks):

        # Execute the query for the chunk
        df_chunk = apply_graphy_features(df_chunk, f"SLOT_{slot_year}")

        # Store the processed chunk in a CSV
        df_chunk.to_csv(f'df_split_{slot_year}_part_{idx + 1}.csv', index=False)
        print(df_chunk.sample(5, random_state=42))

        # Concatenate the results
        df_results = pd.concat([df_results, df_chunk], ignore_index=True)

else:

    # Execute the query
    df_results = apply_graphy_features(df_split_year, f"SLOT_{slot_year}")
    print(df_results.sample(5, random_state=42))

# Store the year split in a csv and in sqllite database
df_results.to_csv(f'df_split_{slot_year}_2.csv', index=False)

conn = sqlite3.connect('split_dataset.db')
df_results.to_sql('split', conn, index=False, if_exists='replace')
conn.close()


# Build train-test splits:
def build_splits():
    # List to store dataFrames from each database
    dfs = []
    # Generate 5 train and test splits from the year datafiles
    for year in range(2010, 2024):
        df_year = pd.read_csv(f'df_split_{year}_2.csv')
        dfs.append(df_year)
    df_all = pd.concat(dfs, ignore_index=True)

    # Split the concatenated DataFrame into train and test sets
    train_set_1 = df_all[df_all['year'].between(2010, 2018)].dropna()
    test_set_1 = df_all[df_all['year'].between(2011, 2019)].dropna()

    train_set_2 = df_all[df_all['year'].between(2010, 2019)]
    test_set_2 = df_all[df_all['year'].between(2011, 2020)].dropna()

    train_set_3 = df_all[df_all['year'].between(2010, 2020)]
    test_set_3 = df_all[df_all['year'].between(2011, 2021)].dropna()

    train_set_4 = df_all[df_all['year'].between(2010, 2021)]
    test_set_4 = df_all[df_all['year'].between(2011, 2022)].dropna()

    train_set_5 = df_all[df_all['year'].between(2010, 2022)]
    test_set_5 = df_all[df_all['year'].between(2011, 2023)].dropna()

    return [(train_set_1, test_set_1), (train_set_2, test_set_2), (train_set_3, test_set_3),
            (train_set_4, test_set_4), (train_set_5, test_set_5)]

splits = build_splits()

# Set the sequence length
seq_length = 8 + slotnum

# Run the model
def run_ts_model(train_set, test_set, seq_length, slotnum):
    # Group the train set by node1 and node2 pairs
    train_grouped = train_set.groupby(['node1', 'node2'])
    test_grouped = test_set.groupby(['node1', 'node2'])


    # Create sequences for each node1 and node2 pair in the train set
    train_sequences = []
    for _, group in train_grouped:
        if len(group) == seq_length:  # Check if there are enough data points for a sequence
            for i in range(len(group) - seq_length + 1):
                sequence = group.iloc[i:i + seq_length]
                train_sequences.append(sequence)

    # Convert the sequences to a dataframe and column year to datetime format
    df_sequences = pd.concat(train_sequences, ignore_index=True)
    df_sequences['year'] = pd.to_datetime(df_sequences['year'], format='%Y')

    # Store train sequence in csv and sql
    df_sequences.to_csv(f'sequences_dataset_{slotnum}.csv', index=False)
    conn = sqlite3.connect(f'sequences_dataset{slotnum}.db')
    df_sequences.to_sql(f'sequences{slotnum}', conn, index=False, if_exists='replace')
    conn.close()

    # Create sequences for each node1 and node2 pair in the test set
    test_sequences = []
    for _, group in test_grouped:
        if len(group) == seq_length:  # Check if there are enough data points for a sequence
            for i in range(len(group) - seq_length + 1):
                sequence = group.iloc[i:i + seq_length]
                test_sequences.append(sequence)

    # Convert the sequences to a dataframe and column year to datetime format
    test_sequences = pd.concat(test_sequences, ignore_index=True)
    test_sequences['year'] = pd.to_datetime(test_sequences['year'], format='%Y')

    # Store train sequence in csv and sql
    test_sequences.to_csv(f'test_sequences_dataset_{slotnum}.csv', index=False)
    conn = sqlite3.connect(f'test_sequences_dataset_{slotnum}.db')
    test_sequences.to_sql(f'test_sequences_{slotnum}', conn, index=False, if_exists='replace')
    conn.close()

    # Convert to numpy type and shift the target data one line downwards so the next time step is the target
    train_features = df_sequences[['cn', 'pa', 'aa', 'tn', 'ra']].to_numpy()
    train_target = df_sequences[['cn', 'pa', 'aa', 'tn', 'ra']].shift(-1).to_numpy()
    test_features = test_sequences[['cn', 'pa', 'aa', 'tn', 'ra']].to_numpy()
    test_target = test_sequences[['cn', 'pa', 'aa', 'tn', 'ra']].shift(-1).to_numpy()

    # Scale the features using a StandardScaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(train_features)
    scaled_test_features = scaler.transform(test_features)

    # Scale the target variables using a different Scaler, since timestep is shifted
    target_scaler = StandardScaler()
    scaled_target = target_scaler.fit_transform(train_target)
    scaled_test_target = target_scaler.transform(test_target)

    # Determine the total number of samples
    num_samples = len(train_target) - seq_length

    # Reshape train features to num_samples size
    train_features = scaled_features[:num_samples].reshape(-1, seq_length, 5)
    train_target = scaled_target[:num_samples].reshape(-1, seq_length, 5)

    # # Reshape test features to num_samples size
    test_features = scaled_test_features[:num_samples].reshape(-1, seq_length, 5)
    test_target = scaled_test_target[:num_samples].reshape(-1, seq_length, 5)

    # Print the shape to check if they are equal
    print("Shape of train features:", train_features.shape)
    print("Shape of train target:", train_target.shape)

    print("Shape of test features:", test_features.shape)
    print("Shape of test target:", test_target.shape)

    # Set the number of features and a batch size that fits the model best
    num_features = 5
    batch_size = 32

    # Use a sequential model and LSTM with 32 units, linear activation and a linear output layer, Dropout of 0.2 is
    # used to enhance generalization
    # model = Sequential([
    #     LSTM(32, activation='linear', return_sequences=True, input_shape=(seq_length, num_features)),
    #     Dropout(0.2),
    #     Dense(5, activation='linear')
    # ])

    # Test a bidirectional LSTM with the same configuration
    # model = Sequential([
    #     Bidirectional(LSTM(32, activation='linear', return_sequences=True), input_shape=(seq_length, num_features)),
    #     Dropout(0.2),
    #     Dense(5, activation='linear')
    # ])

    # Test a GRU model with the same configuration
    model = Sequential([ # Add BatchNormalization for input normalization
        GRU(32, activation='linear', return_sequences=True, input_shape=(seq_length, num_features)),
        Dropout(0.2),
        Dense(5, activation='linear')  # Use a linear activation for regression tasks
    ])

    # Set a fitting learning rate, using Adam optimizer, learning rate should be quite small
    learning_rate = 0.001

    # Compile the model with the Adam optimizer and specified learning rate
    optimizer = legacy.Adam(learning_rate=learning_rate)

    # Compile the model and take mean squared error as loss function
    model.compile(optimizer=optimizer, loss='mse')

    # Train the model using train featuers and targets, epochs can be set accordingly, verbose states what the
    # output of the model fit is and save the model
    history = model.fit(train_features, train_target, epochs=200, verbose=1, batch_size=batch_size)
    model.save(f'model_ts_{slotnum}.keras')

    # Evaluate the model on the test set using the test features and target
    loss = model.evaluate(test_features, test_target, verbose=1)
    print(f'Test Loss in slot {slotnum}: {loss}')

    # Plot the model loss
    plt.plot(history.history['loss'], label='Model Loss')
    plt.title(f'Model Loss in slot {slotnum}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # Make predictions on the test set
    predictions = model.predict(test_features, batch_size=batch_size)

    # Reshape predictions to 2D array again, before was a 3D array
    predictions_2d = predictions.reshape(-1, predictions.shape[-1])

    # Inverse transform the scaled predictions
    original_predictions = target_scaler.inverse_transform(predictions_2d)
    print(len(original_predictions))

    # Also reshape the test target to 2D for comparison
    test_target_flat = test_target.reshape(-1, test_target.shape[-1])

    return test_target_flat, original_predictions, test_sequences

# Get training set and test set from the splits function above. The index has to be reduced by one.
train_set, test_set = splits[slotnum - 1]
# Execute the model training
test_target_flat, original_predictions, test_sequences = run_ts_model(train_set, test_set, seq_length, slotnum)

# Calculate RSME, R2 and MAE for the whole model
def calculate_rsme(test_target_flat, original_predictions, test_sequences, slotnum):
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test_target_flat, original_predictions))
    print(f'RMSE in slot {slotnum}: {rmse}')

    # Calculate R2 score
    r2 = r2_score(test_target_flat, original_predictions)
    print(f'R2 Score in slot {slotnum}: {r2}')

    # Calculate MAE score
    mae = mean_absolute_error(test_target_flat, original_predictions)
    print(f'MAE Score in slot {slotnum}: {mae}')

    # Plot prediction and actual target data for all 10 of the node pairs
    first_10_node_pairs = test_sequences[['node1', 'node2']].drop_duplicates().head(10)
    test_line_colors = []
    # Iterate over each node pair and plot the features
    for index, row in first_10_node_pairs.iterrows():
        node1, node2 = row['node1'], row['node2']

        # Create a new plot for each node pair
        plt.figure()
        plt.title(f'Node Pair: {node1}-{node2} in slot {slotnum}')

        # Filter the dataframe for the current node pair in the test sequences
        current_test_node_pair = test_sequences[(test_sequences['node1'] == node1) & (test_sequences['node2'] == node2)]

        # Get the indices of the current node pair in the original dataframe
        start_index = current_test_node_pair.index.min()
        end_index = current_test_node_pair.index.max() + 1

        # Filter the predictions for the current node pair
        current_predictions = original_predictions[start_index:end_index]

        # Plot the test targets for the current node pair
        for feature in ['cn', 'pa', 'aa', 'tn', 'ra']:
            test_line = plt.plot(current_test_node_pair['year'], current_test_node_pair[feature],
                                 label=f'Test Targets - {feature}')
            test_line_colors.append(test_line[0].get_color())

        # Plot the predicted values for the current node pair
        for feature_index, feature in enumerate(['cn', 'pa', 'aa', 'tn', 'ra']):
            color = test_line_colors[feature_index]
            plt.plot(current_test_node_pair['year'], current_predictions[:, feature_index], linestyle='--',
                     label=f'Predictions - {feature}', color=color)

        # Add legend and labels
        plt.legend(loc='upper left')
        plt.xlabel('Year')
        plt.ylabel('Feature Value')
        plt.show()

    # Calculate RMSE and R2 for each feature
    rmse_per_feature = {}
    r2_per_feature = {}
    mae_per_feature = {}

    for i, feature_name in enumerate(['cn', 'pa', 'aa', 'tn', 'ra']):
        feature_predictions = original_predictions[:, i]
        feature_test_target = test_target_flat[:, i]

        # Calculate RMSE for each feature
        rmse_per_feature[feature_name] = np.sqrt(mean_squared_error(feature_test_target, feature_predictions))

        # Calculate R2 for each feature
        r2_per_feature[feature_name] = r2_score(feature_test_target, feature_predictions)

        # Calculate MAE for each feature
        mae_per_feature[feature_name] = mean_absolute_error(feature_test_target, feature_predictions)

    # Print RMSE, R2 and MAE for each feature
    for feature_name in ['cn', 'pa', 'aa', 'tn', 'ra']:
        print(f'Feature: {feature_name}')
        print(f'  RMSE: {rmse_per_feature[feature_name]}')
        print(f'  R2: {r2_per_feature[feature_name]}')
        print(f'  MAE: {mae_per_feature[feature_name]}')

# Execute the metrics and plots
calculate_rsme(test_target_flat = test_target_flat, original_predictions=original_predictions,
              test_sequences=test_sequences, slotnum=slotnum)

