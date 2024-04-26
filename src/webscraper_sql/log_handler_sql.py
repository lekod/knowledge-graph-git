import csv
import sqlite3


# A data logger is created to capture bugs
class DataLogger:
    def __init__(self, file_name):
        self.file_name = file_name

    def create_logger(self):
        with open(self.file_name, 'w', newline='', encoding='utf-8') as file:
            file.write("Log file created successfully")

    def log_data(self, data):
        with open(self.file_name, 'a') as file:
            file.write(data + '\n')
        print("Data logged successfully.")


logger = DataLogger('data_log.txt')


# This small function just opens the input file once to fill 'sc_address_counter'
def process_address_data_from_file(input_filename):
    open(input_filename, newline='')


# A final Counter is created to finally count every address in the dataset to tell which one is referenced often
class FinalCounter:
    def process_final_data_from_file(self, input_filename, output_filename):
        with open(input_filename, newline='') as f:
            reader = csv.reader(f)
            address_data = [j for i in reader for j in i]

        # Connect to a SQLite database
        conn = sqlite3.connect(output_filename)
        cursor = conn.cursor()

        # Create a SQLite table to store the data if it does not exist
        cursor.execute('''
                    CREATE TABLE IF NOT EXISTS address_counts (
                        address TEXT PRIMARY KEY,
                        count INTEGER
                    )
                ''')

        # Count all the unique addresses together
        for un_address in set(address_data):
            address_count = address_data.count(un_address)
            cursor.execute('''
                            INSERT INTO address_counts (address, count) VALUES (?, ?)
                        ''', (un_address, address_count))

        # Commit the transaction
        conn.commit()

        # Close the database connection
        conn.close()
