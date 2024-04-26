import sqlite3
from urllib import parse
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from scrape_text_sql import ScrapeText
from concurrent.futures import ThreadPoolExecutor
import csv
from log_handler_sql import process_address_data_from_file
from log_handler_sql import logger
import yaml
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)


# Initialize the class
class ScrapeLinks:
    def __init__(self):
        # Set a round counter to check how deep website depth and configure the table name of the SQLite table
        self.round_counter = 0
        self.table_name = config['TABLE_NAME']

    def scrape_links(self):
        # Initialize lists that contain first and second websites if they are already scraped
        already_scraped_fi = set([])
        already_scraped_sc = set([])

        # Set the list with the base addresses you want to scrape
        base_address_list = config['BASE_ADDRESS_LIST']
        address_list = base_address_list

        # Set a list for new addresses
        new_address_list = set([])

        # Connect to SQLite 3 database
        conn = sqlite3.connect(config['DATABASE_NAME'])
        cursor = conn.cursor()

        # Check if the table already exists or configure new table
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {self.table_name} (
        id INTEGER PRIMARY KEY,
        first_website TEXT,
        first_title TEXT,
        second_website TEXT,
        second_title TEXT,
        date DATE,
        time TIME,
        text TEXT
        )
        ''')
        print(f"Table {config['TABLE_NAME']} created successfully.")

        # Begin to scrape the websites with multithreading using the ThreadPoolExecutor
        while len(address_list) > 0:
            with ThreadPoolExecutor() as executor:
                self.round_counter += 1
                # Update the round_counter up to a certain website depth. If depth is reached: stop the scraper
                if self.round_counter == 5:
                    logger.log_data(f"Scraping process is finished: 5 rounds done")
                    break
                logger.log_data(f"round begins: {self.round_counter}")

                # As long as scraper is not stopped: send a request of every base_address and later other address to
                # the ThreadPoolExecutor to scrape their underlying websites, check lists to prevent duplicates
                for address in address_list:
                    executor.submit(self.scrape_address, address, new_address_list, already_scraped_fi,
                                    already_scraped_sc, base_address_list)
                executor.shutdown()

                # Update address_list after execution
                address_list = [x for x in new_address_list if
                                x.startswith(tuple(base_address_list)) and x not in already_scraped_fi]
                pass

        logger.log_data(f"Scraping process is finished: All rounds done")

    # In this function, addresses are scraped and updated and website titles are found
    def scrape_address(self, address, new_address_list, already_scraped_fi, already_scraped_sc, base_address_list):
        sc_ref_count = []
        # Connect to the database again
        conn = sqlite3.connect(config['DATABASE_NAME'])
        cursor = conn.cursor()
        try:
            # Send request to the page, set encoding and use BeautifulSoup library to parse the html text
            base_page = requests.get(address, timeout=config['TIMEOUT'])
            base_page.encoding = 'UTF-8'
            first_page = BeautifulSoup(base_page.text, 'html.parser')
            already_scraped_fi.update({address})
            # The duplicates list ist updated and all links on the page are saved that are stored under html tag 'a'
            all_links = first_page.findAll('a')
            logger.log_data(f"first_page in scrape_address: {address}")

            # For every link in the container of links: construct a clean link first starting with https or www.
            for link in all_links:
                if link.has_attr('href'):
                    cl_link = link['href']
                    if cl_link.startswith("/"):
                        second_page = parse.urljoin(address, cl_link)
                    elif cl_link.startswith("https"):
                        second_page = cl_link
                    elif cl_link.startswith("www."):
                        second_page = "https://" + cl_link
                    elif cl_link.startswith("http://"):
                        second_page = cl_link.replace("http://", "https://", 1)
                    else:
                        continue

                    # Get the new address from the scraping process if not already scraped
                    if second_page not in already_scraped_sc and second_page.startswith(tuple(base_address_list)):
                        new_address_list.update({second_page})
                        scrape_text_instance = ScrapeText(second_page)
                        # request text scraping from the scrape_text_sql file
                        text_list = scrape_text_instance.scrape_text()
                        date_list = scrape_text_instance.scrape_date()
                        already_scraped_sc.update({second_page})
                        logger.log_data(f"scraped sc_link {second_page}")

                    else:
                        # Handle duplicates: append them to the duplicates list and add it to the address counter
                        sc_ref_count.append(second_page)
                        with open("sc_address_counter", 'a', newline='', encoding='utf-8') as csvfile:
                            csv_writer = csv.writer(csvfile)
                            csv_writer.writerow([second_page])

                            input_file = 'sc_address_counter'
                            process_address_data_from_file(input_file)
                        continue

                else:
                    continue

                # Identify the title of the website of the first page
                first_title = first_page.title.text.strip() if first_page.find('title') else logger.log_data(
                        "something wrong with first_page: title")

                if link.find('title'):
                    second_title = link.title.text.strip()
                elif link.text:
                    second_title = link.text.strip()
                elif link.has_attr('title'):
                    second_title = link['title'].strip()
                else:
                    try:
                        second_title = link.string
                    except Exception as e:
                        logger.log_data(f"something wrong with second_page: title: {e}")
                        second_title = " "

                # Add the time the link was scraped to the csv to monitor the progress
                scraped_time = datetime.now().strftime("%H:%M:%S")

                # Check if the text extracted from the website contains at least 20 characters, so it is actually a text
                if any(len(word) >= 20 and any(char.isalpha() for char in word) for word in text_list):
                    text_str = ', '.join(text_list)

                    # Write all information together in the SQLite3 table and commit that
                    cursor.execute(f'''
                                     INSERT INTO {self.table_name} (first_website, first_title, second_website,
                                     second_title, date, time, text)
                                     VALUES (?, ?, ?, ?, ?, ?, ?)
                                 ''',
                                   (str(address), str(first_title), str(second_page), str(second_title),
                                    str(date_list), str(scraped_time), text_str))
                    conn.commit()
                else:
                    continue

        # Log data if problems occur
        except Exception as e:
            logger.log_data(f"Problems with scraping occur: {e} with address: {address}")
