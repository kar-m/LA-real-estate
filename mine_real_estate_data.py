import sys
from selenium import webdriver
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import time

# Function to scrape pages form the realtor.com website
def scrape_page(cards, prices, beds, baths, house_size, lot_size, zip):
  for card in cards:
    # All code blocks are in try-except batches, because of irregularities in the website's formatting
    # Scraping house prices
    try:
      # Collecting the string value from the relevant attribute
      a = (card.find('span', attrs={"data-label":"pc-price"}).text)
      
      # Cleaning up the string to make it into an int
      a = a.replace("$", "")
      a = a.replace(",", "")
      
      # Converting the clean string into an int
      a = int(a)
      prices.append(a)
      
    except:
      # Appening a nan into the list to keep all lists of values aligned
      prices.append(np.nan)


    # Scraping number of beds in a house
    try:
      bed_container = card.find('li', attrs={"data-label":"pc-meta-beds"})
      beds.append(int(bed_container.find('span', attrs={"data-label":"meta-value"}).text))

    except:
      beds.append(np.nan)


    bath_container = card.find('li', attrs={"data-label":"pc-meta-baths"})
    try:
      a = (bath_container.find('span', attrs={"data-label":"meta-value"}).text)
      a = a.replace("+", "")
      a = float(a)
      baths.append(a)

    except:
      baths.append(np.nan)

    sqft_container = card.find('li', attrs={"data-label":"pc-meta-sqft"})
    try:
      a = (sqft_container.find('span', attrs={"data-label":"meta-value"}).text)
      a = a.replace(",", "")
      a = int(a)
      house_size.append(a)

    except:
      house_size.append(np.nan)

    lot_container = card.find('li', attrs={"data-label":"pc-meta-sqftlot"})
    try:
      a = (lot_container.find('span', attrs={"data-label":"meta-value"}).text)
      a = a.replace(",", "")
      a = float(a)

      size_type = lot_container.find('span', attrs={"data-label":"meta-label"}).text
      if 'acre' in size_type:
        a *= 43650

      lot_size.append(a)

    except:
      lot_size.append(np.nan)

    try:
      zip.append(card.find('div', attrs={"data-label":"pc-address-second"}).text)
  
    except:
      zip.append(np.nan)

prices = []
beds = []
baths = []
lot_size = []
house_size = []
zip = []

url = "https://www.realtor.com/realestateandhomes-search/Los-Angeles_CA"

try:
  wd.quit()
except:
  pass

# Initializing Selenium for scraping
wd = webdriver.Chrome('chromedriver')
wd.get(url)
content = wd.page_source

# Initializing BeautifulSoup parser
soup = BeautifulSoup(content, features='html.parser')
cards = soup.findAll('li', attrs={"data-testid": "result-card"})

scrape_page(cards, prices, beds, baths, house_size, lot_size, zip)

# Defining all the url-s of interest according to how realtor.com indexes pages
urls = ["https://www.realtor.com/realestateandhomes-search/Los-Angeles_CA/pg-" + str(i) for i in range(2, 207)]


for url in urls:
    # Provide breaks between website pings to help realtor.com's SysAdmins sleep at night :)
    time.sleep(1)
    wd.get(url)
    content = wd.page_source
    
    soup = BeautifulSoup(content, features='html.parser')
    cards = soup.findAll('li', attrs={"data-testid": "result-card"})
    scrape_page(cards, prices, beds, baths, house_size, lot_size, zip)
    
    if len(cards) == 0:
        time.sleep(15)
  
# Save generated data
df = pd.DataFrame({'BedNumber': beds, 'BathNumber': baths, 'HouseSize': house_size, 'LotSize': lot_size, 'ZipCode': zip, 'Price': prices})

df.to_csv('Desktop\\real_estate_data.csv')
