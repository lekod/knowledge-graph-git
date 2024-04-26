from scrape_links_sql import ScrapeLinks
from log_handler_sql import logger
from log_handler_sql import FinalCounter
import yaml
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)


# First: configure your data request in the config.yml file, instructions are given
# Run main to fully webscrape the data
# In the code you can find some additions like PDF scraping or setting the depth of the scraping, you still have to set
def main():
    scraper = ScrapeLinks()
    final_counter = FinalCounter()

    # Call scraper
    scraper.scrape_links()
    logger.log_data("scraping data is finished, going to Final Counter")

    # Call final address counter for counting how many times a page was visited
    logger.log_data("final_counter processing started")
    final_counter.process_final_data_from_file(config['INPUT_FILE'], config['OUTPUT_FILE'])
    logger.log_data("sc_counter_list is finished")


if __name__ == "__main__":
    main()
