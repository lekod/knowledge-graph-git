import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sqlite3

# Fill out all the RSME values for every of the 5 splits of each model
rmse_lstm = [8.16362, 9.31013, 9.142035, 8.89995, 9.07919]
rmse_bidi_lstm = [8.70714, 9.30205, 9.51715, 10.13230, 10.44411]
rmse_gru = [8.93470, 8.94121, 8.77506, 8.97465, 8.18792]

# Calculate the mean, standard error and confidence interval of each result
# For RSME: the smaller the mean, the better
def calculate_confidence_interval(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    std_error = std_dev / np.sqrt(len(data))
    z_score = stats.norm.ppf(0.975)  # 95% confidence level
    lower_bound = mean - z_score * std_error
    upper_bound = mean + z_score * std_error
    return "{:.6f}".format(mean), "{:.6f}".format(std_error), ("{:.6f}".format(lower_bound), "{:.6f}".format(upper_bound))

mean_lstm, std_error_lstm, ci_lstm = calculate_confidence_interval(rmse_lstm)
mean_bidi, std_error_bidi, ci_bidi = calculate_confidence_interval(rmse_bidi_lstm)
mean_gru, std_error_gru, ci_gru = calculate_confidence_interval(rmse_gru)

# Print results
print("LSTM:")
print("  Mean:", mean_lstm)
print("  Standard Error:", std_error_lstm)
print("  95% Confidence Interval:", ci_lstm)
print()
print("Bidirectional LSTM:")
print("  Mean:", mean_bidi)
print("  Standard Error:", std_error_bidi)
print("  95% Confidence Interval:", ci_bidi)
print()
print("GRU:")
print("  Mean:", mean_gru)
print("  Standard Error:", std_error_gru)
print("  95% Confidence Interval:", ci_gru)

############################## VALIDATION ################################

# Build a separate validation split between 2011 and 2024
def build_splits_val():
    dfs = []  # List to store DataFrames from each database
    for year in range(2011, 2025):
        df_year = pd.read_csv(f'df_split_{year}_2.csv')  # Replace with your database file name
        dfs.append(df_year)
    df_all = pd.concat(dfs, ignore_index=True)

    # Validation split
    train_set_6 = df_all[df_all['year'].between(2011, 2023)]
    test_set_6 = df_all[df_all['year'].between(2012, 2024)].dropna()

    return [(train_set_6, test_set_6)]

splits = build_splits_val()

# Run the predefined model on the validation set
def run_val_model(train_set, test_set, seq_length, slotnum):

    # Group the train set by node1 and node2 pairs
    train_grouped = train_set.groupby(['node1', 'node2'])
    test_grouped = test_set.groupby(['node1', 'node2'])

    # Create sequences for each node1 and node2 pair in the validation set
    sequences = []
    for _, group in train_grouped:
        if len(group) == seq_length:  # Check if there are enough data points for a sequence
            for i in range(len(group) - seq_length + 1):
                sequence = group.iloc[i:i + seq_length]
                sequences.append(sequence)

    # Convert the sequences to a dataframe and to csv and sql database
    df_sequences = pd.concat(sequences, ignore_index=True)
    # Convert year to datetime format
    df_sequences['year'] = pd.to_datetime(df_sequences['year'], format='%Y')
    df_sequences.to_csv(f'sequences_dataset_{slotnum}.csv', index=False)
    conn = sqlite3.connect(f'sequences_dataset{slotnum}.db')
    df_sequences.to_sql(f'sequences{slotnum}', conn, index=False, if_exists='replace')
    conn.close()

    # Create test sequences
    test_sequences = []
    for _, group in test_grouped:
        if len(group) == seq_length:  # Check if there are enough data points for a sequence
            for i in range(len(group) - seq_length + 1):
                sequence = group.iloc[i:i + seq_length]
                test_sequences.append(sequence)

    test_sequences = pd.concat(test_sequences, ignore_index=True)
    # Convert year to datetime format
    test_sequences['year'] = pd.to_datetime(test_sequences['year'], format='%Y')
    # Convert the sequences to a dataframe and to csv and sql database
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

    # Scale the target variables
    target_scaler = StandardScaler()
    scaled_target = target_scaler.fit_transform(train_target)
    scaled_test_target = target_scaler.transform(test_target)

    # Determine the total number of samples
    num_samples = len(train_target) - seq_length

    # Reshape train features
    train_features = scaled_features[:num_samples].reshape(-1, seq_length, 5)
    train_target = scaled_target[:num_samples].reshape(-1, seq_length, 5)

    # Reshape test features
    test_features = scaled_test_features[:num_samples].reshape(-1, seq_length, 5)
    test_target = scaled_test_target[:num_samples].reshape(-1, seq_length, 5)

    print("Shape of features Validation:", train_features.shape)
    print("Shape of target Validation:", train_target.shape)

    print("Shape of test features Validation:", test_features.shape)
    print("Shape of test target Validation:", test_target.shape)

    # load the pretrained model and fit it
    model = load_model('final_model.keras')

    batch_size = 32
    history = model.fit(train_features, train_target, epochs=300, verbose=1, batch_size=batch_size)

    # Evaluate the model on the test set
    loss = model.evaluate(test_features, test_target, verbose=1)
    print(f'Test Loss in slot {slotnum} Validation: {loss}')

    # Plot training and test loss
    plt.plot(history.history['loss'], label='Model Loss')
    plt.title(f'Model Loss in slot {slotnum} Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # Make predictions on the test set
    predictions = model.predict(test_features, batch_size=batch_size)

    # Reshape predictions to 2D array
    predictions_2d = predictions.reshape(-1, predictions.shape[-1])

    # Inverse transform the scaled predictions
    original_predictions = target_scaler.inverse_transform(predictions_2d)
    print(len(original_predictions))

    # Also reshape the test target to 2D for comparison
    test_target_flat = test_target.reshape(-1, test_target.shape[-1])

    return test_target_flat, original_predictions, test_sequences

# Execute validation splits
val_splits = build_splits_val()

# configure to use the validation slot, but the seq_length of the train split 6
slotnum = 6
seq_length = 13

# Execute the model prediction function
train_set, test_set = val_splits[0]
test_target_flat, original_predictions, test_sequences = run_val_model(train_set, test_set, seq_length, slotnum)

# Calculate RSME, R2 and MAE for the whole model
def calculate_rsme(test_target_flat, original_predictions, test_sequences, slotnum):
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(test_target_flat, original_predictions))
    print(f'RMSE in slot {slotnum} Validation: {rmse}')

    # Calculate R2 score
    r2 = r2_score(test_target_flat, original_predictions)
    print(f'R2 Score in slot {slotnum} Validation: {r2}')

    # Calculate MAE score
    mae = mean_absolute_error(test_target_flat, original_predictions)
    print(f'MAE Score in slot {slotnum} Validation: {mae}')

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

        # Filter the predictions for each node pair
        current_predictions = original_predictions[start_index:end_index]

        # Plot the test targets for each node pair
        for feature in ['cn', 'pa', 'aa', 'tn', 'ra']:
            test_line = plt.plot(current_test_node_pair['year'], current_test_node_pair[feature],
                                 label=f'Test Targets - {feature}')
            test_line_colors.append(test_line[0].get_color())

        # Plot the predicted values for each node pair
        for feature_index, feature in enumerate(['cn', 'pa', 'aa', 'tn', 'ra']):
            color = test_line_colors[feature_index]
            plt.plot(current_test_node_pair['year'], current_predictions[:, feature_index], linestyle='--',
                     label=f'Predictions - {feature}', color=color)

        # Add legend and labels
        plt.legend(loc='upper left')
        plt.xlabel('Year')
        plt.ylabel('Feature Value')
        plt.show()

    # Calculate RMSE, R2 and MAE for each feature
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
        print(f'  RMSE Validation: {rmse_per_feature[feature_name]}')
        print(f'  R2 Validation: {r2_per_feature[feature_name]}')
        print(f'  MAE Validation: {mae_per_feature[feature_name]}')

# Execute the metrics calculations and plots
calculate_rsme(test_target_flat = test_target_flat, original_predictions=original_predictions,
               test_sequences=test_sequences, slotnum=slotnum)




############################### FINAL PREDICTION ####################################


def build_splits_pred():
    dfs = []  # List to store DataFrames from each database
    for year in range(2012, 2025):
        df_year = pd.read_csv(f'df_split_{year}_2.csv')  # Replace with your database file name
        dfs.append(df_year)
    df_all = pd.concat(dfs, ignore_index=True)

    # Prediction split
    train_set_7 = df_all[df_all['year'].between(2012, 2024)]

    return [train_set_7]

pred_split = build_splits_pred()

def run_pred_model(train_set, seq_length, slotnum):
    # Group the train set by node1 and node2 pairs
    train_grouped = train_set.groupby(['node1', 'node2'])
    print(len(train_grouped))


    # Create sequences of length 9 for each node1 and node2 pair in the train set
    sequences = []
    for _, group in train_grouped:
        if len(group) == seq_length:  # Check if there are enough data points for a sequence
            for i in range(len(group) - seq_length + 1):
                sequence = group.iloc[i:i + seq_length]
                sequences.append(sequence)

    # Convert the sequences to a DataFrame
    df_sequences = pd.concat(sequences, ignore_index=True)
    df_sequences['year'] = pd.to_datetime(df_sequences['year'], format='%Y')

    # Save sequences to CSV
    df_sequences.to_csv(f'sequences_dataset_{slotnum}.csv', index=False)

    # Save sequences to SQLite database
    conn = sqlite3.connect(f'sequences_dataset{slotnum}.db')
    df_sequences.to_sql(f'sequences{slotnum}', conn, index=False, if_exists='replace')
    conn.close()
    #
    # Print the first few rows of the train and test sets, and the sequences DataFrame
    print("\nTrain Sequences prediction:")
    print(df_sequences.head())

    train_features = df_sequences[['cn', 'pa', 'aa', 'tn', 'ra']].to_numpy()
    # Scale the data as in the training
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(train_features)
    train_features = scaled_features.reshape(-1, seq_length, 5)
    print("Shape of features Prediction:", train_features.shape)

    # Set the Batch size similar to the training batch and load the final model ( GRU split 5)
    batch_size = 32
    model = load_model('final_model.keras')

    # Make predictions using the prediction train features
    predictions = model.predict(train_features, batch_size=batch_size)

    # Reshape predictions to 2D array
    predictions_2d = predictions.reshape(-1, predictions.shape[-1])

    # Inverse transform the scaled predictions
    original_predictions = scaler.inverse_transform(predictions_2d)
    print(len(original_predictions))

    pd.DataFrame(original_predictions).to_csv(f'final_prediction.csv', index=False)

    # Combine original predictions with node1 and node2 columns
    df_original_predictions = df_sequences[['year', 'node1', 'node2', 'label']].copy()
    df_original_predictions['year'] = df_original_predictions['year'] + pd.DateOffset(years=1)

    df_original_predictions[['cn', 'pa', 'aa', 'tn', 'ra']] = original_predictions

    # Reset index before saving to SQLite database
    df_original_predictions.reset_index(drop=True, inplace=True)

    # Save original predictions to SQLite database
    conn = sqlite3.connect(f'final_prediction.db')
    df_original_predictions.to_sql(f'final prediction', conn, index=False, if_exists='replace')
    conn.close()

val_splits = build_splits_val()

# The slotnumber and sequence length are already set since we want to use the prediction split
slotnum = 7
seq_length = 13
train_set = pred_split[0]

original_predictions = run_pred_model(train_set, seq_length, slotnum)













