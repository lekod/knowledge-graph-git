import requests
import io
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import re
from htmldate import find_date
import yaml
from log_handler_sql import logger
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)


# initialize the text scraping class
class ScrapeText:
    def __init__(self, cl_link):
        # Initialize clean link variable
        self.cl_link = cl_link

    def scrape_text(self):
        # Exclude all files that do not contain text or that should not be investigated
        endings = (".jpg", ".jpeg", ".png", ".xls", ".xlsx", ".doc", ".docx", ".zip", ".rss", ".mp3", ".mp4")
        if not self.cl_link.endswith(endings):
            # Request the page if it is not an excluded file link and give a timeout with it
            page = requests.get(self.cl_link, timeout=config['TIMEOUT'])

            # PDFs are read differently with a PdfReader package and
            if self.cl_link.endswith(".pdf"):
                try:
                    f = io.BytesIO(page.content)
                    pdf = PdfReader(f)
                    # Clean out certain characters
                    texts_list_cl = [re.sub(r'[\n\xa0\r\xad\t\x80\x9e\x9c\uf0d8]', ' ', pagenum.extract_text())
                                     for pagenum in pdf.pages]
                    return texts_list_cl
                except (Exception,):
                    texts_list_cl = " "
                    return texts_list_cl

            else:
                # Normal websites are just encoded by UTF-8 and BeautifulSoup package is used to extract the text
                page.encoding = 'UTF-8'
                page_soup = BeautifulSoup(page.text, 'html.parser')
                texts_list_cl = [re.sub(r'[\n\xa0\x00\x07\r\xad\t\x80\x9e\x9c]', ' ', text.get_text())
                                 for text in page_soup.find_all('p')]
                return texts_list_cl   # Return the list of texts after scraping

    # Another function is used to scrape the date od the website using the content of the website that gives us the
    # last-updated date using unicode-escape encoding. The date is returned to the scrape_address function
    def scrape_date(self):
        try:
            date_str = requests.get(self.cl_link, timeout=config['TIMEOUT']).content.decode('unicode-escape')
            date = find_date(date_str)
        except Exception as e:
            logger.log_data(f"no date: {e}")
            date = []
        return date
