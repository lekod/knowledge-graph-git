import re
import os
import sqlite3
import csv
from concurrent.futures import ThreadPoolExecutor
from flair.data import Sentence
from flair.models import SequenceTagger
from logger_sql import logger
import nltk
# nltk.download('punkt')

# Entity Recognition should be performed to plain text, so we get final organisations and persons stakeholders
class EntityRec:
    def __init__(self):
        # FLAIR NLP is used and loaded via the sequencetagger
        self.tagger = SequenceTagger.load("flair/ner-german-large")
        # We build a uniwue set that contains already processed URLs and a checkpoint file as backup
        self.processed_urls = set()
        self.checkpoint_file = "checkpoint.txt"

    # We multithread the NER once again to use our capacities efficiently
    def multithread_ner(self):
        self.load_checkpoint()
        # Connect to the Database that contains the abbreviation deletion
        source_conn = sqlite3.connect('scraped_data_text_01.db')
        source_cursor = source_conn.cursor()

        # Construct a destination SQL databse to store the results of NER
        destination_conn = sqlite3.connect('ner_data.db')
        destination_cursor = destination_conn.cursor()

        destination_cursor.execute('''
                            CREATE TABLE IF NOT EXISTS ner_data (
                                id INTEGER PRIMARY KEY,
                                second_website TEXT,
                                date DATE,
                                persons TEXT,
                               organisations TEXT
                            )
                        ''')
        logger.log_data("ner_data database created successfully.")

        source_cursor.execute('SELECT * FROM scraped_data_abbr')

        # Fetch all rows from the source database
        rows = source_cursor.fetchall()

        # Execute the NER using multithreading
        with ThreadPoolExecutor() as executor:
            logger.log_data("NER is started")
            for row in rows:
                try:
                    # Set the column for the second website, date and test
                    second_website = row[3]
                    date = row[5]
                    text = row[7]
                    cleaned_text = re.sub(r'\s+', ' ', text)
                    # If second_website has not yet been processed, submit the text, website and date to ner_names_orgs
                    # function
                    if second_website not in self.processed_urls:
                        executor.submit(self.ner_names_orgs, cleaned_text, second_website, date)
                except Exception as e:
                    logger.log_data(f"Error processing thread: {e}")
                    continue

    # Function for namen entity recognition
    def ner_names_orgs(self, cleaned_text, second_website, date):
        # List to store extracted personal and organizational data
        pers_data = []
        org_data = []
        try:
            # Tokenize the cleaned text into sentences
            sent_tokens = nltk.sent_tokenize(cleaned_text)
            # Iterate through each sentence
            for i in sent_tokens:
                sent = Sentence(i)
                # Use the tagger to predict named entities
                self.tagger.predict(sent)
                data_list = sent.get_spans('ner')

                # Iterate through each named entity span
                for span in data_list:
                    # Check if the span represents a person
                    if span.labels[0].value == "PER":
                        pers_name = span.text
                        # Check if the name has more than one word
                        if len(pers_name.split()) > 1:
                            # Append the name to the names list
                            pers_data.append(pers_name)
                            with open("pers_counter", 'a', newline='', encoding='utf-8') as csvfile:
                                csv_writer = csv.writer(csvfile)
                                csv_writer.writerow([pers_name])

                    # Check if the span represents an organization
                    elif span.labels[0].value == "ORG":
                        org_name = span.text
                        org_data.append(org_name)
                        with open("org_counter", 'a', newline='', encoding='utf-8') as csvfile:
                            csv_writer = csv.writer(csvfile)
                            csv_writer.writerow([org_name])

            # If personal or organizational data is extracted
            if pers_data or org_data:
                try:
                    destination_conn = sqlite3.connect('ner_data.db')
                    destination_cursor = destination_conn.cursor()

                    # Insert extracted data into the database
                    destination_cursor.execute('''
                                        INSERT INTO ner_data (second_website, date, persons, organisations)
                                        VALUES (?, ?, ?, ?) -- specify parameter placeholders
                                    ''', (second_website, date, str(set(pers_data)), str(set(org_data))))

                    # Commit changes to the database
                    destination_conn.commit()
                    # Update checkpoint
                    self.update_checkpoint(second_website)

                except sqlite3.Error as e:
                    print("Error:", e)
        except MemoryError as mem_error:
            logger.log_data(f"Memory error occurred: {mem_error}")
            pass
        except Exception as e:
            logger.log_data(f"Error processing row: {e}")
            pass

    # Make an organisation counter to know the number of occurrence of organisations
    def org_counter(self, input_filename, output_filename):
        print("org_counter_list.csv is started")
        with open(input_filename, newline='') as f:
            reader = csv.reader(f, delimiter=';')
            org_count_data = [j for i in reader for j in i]

        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=",")
            for i in set(org_count_data):
                address_count = {i: org_count_data.count(i)}         # for i in org_count_data}
                for address, number in address_count.items():
                    csv_writer.writerow([address, number])

        logger.log_data("org_counter_list.csv is finished")

    # Make an personal names counter to know the number of occurrence of names
    def pers_counter(self, input_filename, output_filename):
        print("pers_counter_list.csv is started")
        with open(input_filename, newline='') as f:
            reader = csv.reader(f, delimiter=';')
            pers_count_data = [j for i in reader for j in i]

        with open(output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            pers_count = {i: pers_count_data.count(i) for i in pers_count_data}
            csv_writer = csv.writer(csvfile, delimiter=",")
            for pers, number in pers_count.items():
                csv_writer.writerow([pers, number])

        logger.log_data("org_counter_list.csv is finished")

    def store_processed_urls_to_file(self):
        # Connect to the SQLite database
        conn = sqlite3.connect('ner_data.db')
        cursor = conn.cursor()

        # Execute the SQL query to fetch distinct second_website values
        cursor.execute("SELECT DISTINCT second_website FROM ner_data")
        unique_second_websites = cursor.fetchall()

        # Write the unique values to a text file
        with open('processed_urls.txt', 'w') as file:
            for website in unique_second_websites:
                file.write(website[0] + '\n')

        # Close the database connection
        conn.close()

    # Load and update the checkpoints as backup, since the NER runs a long time
    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, "r") as input_file:
                for line in input_file:
                    self.processed_urls.add(line.strip())
        else:
            with open(self.checkpoint_file, "w"):
                pass

    def update_checkpoint(self, second_website):
        self.processed_urls.add(second_website)
        with open(self.checkpoint_file, "a") as input_file:
            input_file.write(second_website + "\n")