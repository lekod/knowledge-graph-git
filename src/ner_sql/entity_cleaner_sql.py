import re
import sqlite3
import csv
from fuzzywuzzy import fuzz
from spacy import load as spacy_load
from spikex.pipes import AbbrX
from concurrent.futures import ThreadPoolExecutor
from logger_sql import logger
from itertools import product
from collections import defaultdict
import shutil


# First, the abbreviations are replaced through the long version
class Abbreviation:
    def __init__(self):
        # We use spacy's de_core_news_lg for it and set the maximum text length we want to process
        self.nlp = spacy_load("de_core_news_lg")
        self.nlp.max_length = 1000000

    # We use multithreading to use the capacities we have properly
    def multithread_abbr(self):
        # Connect to or construct the source SQLite database
        source_conn = sqlite3.connect('../entity-rec/scraped_data.db')
        source_cursor = source_conn.cursor()

        destination_conn = sqlite3.connect('cleaned_data.db', check_same_thread=False)
        destination_cursor = destination_conn.cursor()

        destination_cursor.execute('''
                    CREATE TABLE IF NOT EXISTS cleaned_data (
                        id INTEGER PRIMARY KEY,
                        first_website TEXT,
                        first_title TEXT,
                        second_website TEXT,
                        second_title TEXT,
                        date DATE,
                        time TIME,
                        text_cleaned TEXT
                    )
                ''')
        print("Table 'cleaned_data' created successfully.")

        # Query to select data from the source database
        source_cursor.execute('SELECT * FROM scraped_data')
        rows = source_cursor.fetchall()
        logger.log_data("file creation of scraped_data_text_01.csv is started")

        # Multithreading for exporting data to the destination database
        with ThreadPoolExecutor() as executor:
            logger.log_data("Abbreviation deletion is started")
            for row in rows:
                executor.submit(self.delete_abbr, row)

        logger.log_data("deleting abbreviations is finished")

    # In the delete_abbr function, the Abbreviations are deleted and replaced
    def delete_abbr(self, row):
        try:
            # Use the text column and call AbrrX function
            text_cleaned = row[7].strip()
            strings_spacy = self.nlp(row[7])
            abbrx = AbbrX(self.nlp.vocab)

            # Configure what is the lond and short form: If longform is actually shorter that short form, it should
            # use the long form instead
            doc = abbrx(strings_spacy)
            for abbr in doc._.abbrs:
                long_form = abbr._.long_form
                short_form = abbr
                if len(str(long_form)) > len(str(short_form)):
                    abbr_rep = long_form
                else:
                    abbr_rep = short_form

                # The replacing of the abbreviation is stored in cleaned text variable
                text_cleaned = text_cleaned.replace(str(short_form), str(abbr_rep))
                cleaned_text = re.sub(r'\s+', ' ', text_cleaned)

            try:
                # Insert the data into the destination database and define the column names and parameters
                destination_conn = sqlite3.connect('cleaned_data.db', check_same_thread=False)
                destination_cursor = destination_conn.cursor()
                destination_cursor.execute('''
                    INSERT INTO cleaned_data (first_website, first_title, second_website,
                                     second_title, date, time, text_cleaned)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (row[1], row[2], row[3], row[4], row[5], row[6], cleaned_text))
                destination_conn.commit()

            # Check for Errors and Exceptions
            except sqlite3.Error as e:
                print("Error:", e)
        except ZeroDivisionError:
            pass
        except ValueError:
            logger.log_data(f"ValueError occured")
            pass


# In the class Similarity, the similarity of two words are calculated, so we receive duplicates as few as possible.
# This is just used for the stakeholder organisations, since these are more diverse
class Similarity:

    def multithread_sim(self):
        source_conn = sqlite3.connect('ner_data.db')
        source_cursor = source_conn.cursor()

        destination_conn = sqlite3.connect('ner_data_sim.db')
        destination_cursor = destination_conn.cursor()
        source_db_path = 'ner_data.db'
        destination_db_path = 'ner_data_sim.db'

        # Copy the database file
        shutil.copyfile(source_db_path, destination_db_path)
        print('ner_data_sim is constructed')

        # Construct a destination database from a copy of the source database
        conn = sqlite3.connect('ner_data_sim.db')
        cursor = conn.cursor()
        cursor.execute('''
                    ALTER TABLE ner_data RENAME TO ner_data_sim
                ''')

        destination_cursor.execute('''
                    CREATE TABLE IF NOT EXISTS ner_data_sim (
                                id INTEGER PRIMARY KEY,
                                second_website TEXT,
                                date DATE,
                                persons TEXT,
                               organisations TEXT,
                               cleaned TEXT
                    )
                ''')
        print("Table 'ner_data_sim' created successfully.")
        logger.log_data("Make similar is processing")
        source_cursor.execute('SELECT * FROM ner_data')
        cursor.execute('SELECT * FROM ner_data_sim')
        rows = cursor.fetchall()

        # Execute the multithreading of the similarity calculation
        with ThreadPoolExecutor() as executor:
            # Use a set of unique organisations
            whole_orgs = set()
            for row in rows:
                # Use the orgs column and strip the name out of the Bracket style
                orgs = row[4]
                org_names = [entry.strip("'{}").strip("'").strip(" "" ").replace("'", "") for entry in orgs.split(",")]
                whole_orgs.update(org_names)
            for row in rows:
                try:
                    # Submit the whole_orgs list, the row into the make_similar function to multithread
                    executor.submit(self.make_similar, row, whole_orgs)
                except Exception as e:
                    logger.log_data(f"something wrong with row: {e}")

            logger.log_data("Make similar() is finished")


# Use this function to check for similar orgs to combine them
    def make_similar(self, row, whole_orgs):
        # Set threshold quite high to only receive good results
        threshold = 90
        organisations = row[4]
        orgs_names = [entry.strip("'{}'").replace("'", "") for entry in organisations.split(",")]

        # Create a dictionary to store counter information for each organization
        org_counter = defaultdict(int)

        # Populate org_counter with data from 'org_counter_list.csv' that listed all occurrences of the orgs
        # We use this list to define, if we have StMD and StMDs, which one is the original version. We expect
        # that it will be the one with most occurrences
        with open('org_counter_list.csv', 'r') as csvcheck:
            checkreader = csv.reader(csvcheck, delimiter=';')
            for checkrow in checkreader:
                try:
                    check_org, check_count = checkrow[0].rsplit(',', 1)
                    org_counter[check_org.strip()] = int(check_count)
                except Exception:
                    continue

        # Find similar organizations in the dataset
        for org1, org2 in product(orgs_names, whole_orgs):
            # Check the similarity score from fuzzywuzzy package for non-equal orgs
            if org1 != org2:
                similarity_score = fuzz.token_sort_ratio(org1, org2)
                # Find matches that exceed the similarity score of 0.9
                if similarity_score >= threshold:
                    # Set the right version according to the highest occurrence counter
                    if org_counter[org1] >= org_counter[org2]:
                        clean_org = org1
                        print(f"used org1: {org1} instead of org2: {org2}, index:{row[0]}")
                        # Connect to the database
                        destination_conn = sqlite3.connect('ner_data_sim.db')
                        destination_cursor = destination_conn.cursor()

                        # Execute the update statement
                        destination_cursor.execute('''
                                        UPDATE ner_data_sim
                                        SET organisations = REPLACE(organisations, ?, ?)
                                        WHERE organisations LIKE ?
                                    ''', ('org2', str(clean_org), '%org2%'))
                        destination_conn.commit()
                    else:
                        clean_org = org2
                        print(f"used org2: {org2} instead of org1: {org1}, index:{row[0]}")
                        # Connect to the database
                        destination_conn = sqlite3.connect('ner_data_sim.db')
                        destination_cursor = destination_conn.cursor()

                        # Execute the update statement
                        destination_cursor.execute('''
                                        UPDATE ner_data_sim
                                        SET organisations = REPLACE(organisations, ?, ?)
                                        WHERE organisations LIKE ?
                                    ''', ('org1', str(clean_org), '%org1%'))
                        destination_conn.commit()
                    break  # Exit loop if a match is found
