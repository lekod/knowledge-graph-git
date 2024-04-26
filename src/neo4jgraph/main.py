# This main is just needed for execution of the neo4J database building from SQL, statistical analysis is executed
# directly in Neo4J Browser and the link predictions are executed from the scripts.
# Connect to localhost under http://localhost:7001/browser/
# Put in terminal: "docker compose up"
import sqlite3
from neomodel import config, db
from dotenv import load_dotenv
import os
from neomodel import install_all_labels
from queries_sql import (create_base_websites, create_second_websites, rel_firstwebsite_to_secondwebsite,
                         rel_basewebsite_to_secondwebsite, create_stakeholder_names,
                         create_stakeholder_orgs, rel_secondwebsite_to_person, rel_secondwebsite_to_org,
                         rel_person_to_org)

# You need the datafiles scraped_data.db and ner_data_sim.db for execution
load_dotenv()
uri = os.getenv("uri")
config.DATABASE_URL = uri
db.set_connection(config.DATABASE_URL)

install_all_labels()

conn = sqlite3.connect('../entity-rec/scraped_data.db')
cursor = conn.cursor()

create_base_websites(cursor)
create_second_websites(cursor)
rel_basewebsite_to_secondwebsite(cursor)
rel_firstwebsite_to_secondwebsite(cursor)


cursor.close()
conn.close()

conn_stake = sqlite3.connect('../ner_sql/ner_data_sim.db')
cursor_stake = conn_stake.cursor()

create_stakeholder_names(cursor_stake)
create_stakeholder_orgs(cursor_stake)
rel_secondwebsite_to_person(cursor_stake)
rel_secondwebsite_to_org(cursor_stake)
rel_person_to_org(cursor_stake)

cursor_stake.close()
conn_stake.close()


# For statistical analysis, the queries have to be put directly in the Neo4JBrowser
# For link prediction and time-series forecasting, the splits have to be configured first in the config file and
# need to be executed directly in the corresponding scripts






