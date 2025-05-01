from data_downloading_and_scraping.crawl_foreign_exchange_rate.crawl_utils.utils import setup_driver, save_to_csv

import time
from datetime import datetime, timedelta

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from bs4 import BeautifulSoup


def set_date_and_search_bidv(
        driver: webdriver.Chrome,
        date_str: str
    ):
    """Set the date in the input field and click the search button."""
    try:
        date_input = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "filter-by-start-date"))
        )
        date_input.clear()
        date_input.send_keys(date_str)
        time.sleep(1)
        
        search_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "clickSearch"))
        )
        search_button.click()
        time.sleep(3)
        
        print(f"Successfully set date to {date_str} and searched")
        return True
    
    except Exception as e:
        print(f"Error setting date and searching: {e}")
        return False


def extract_exchange_rates_bidv(
        html_content: str,
        date_str: str
    ):
    """Extract exchange rates from the HTML content."""
    soup = BeautifulSoup(html_content, "html.parser")
    table = soup.find("table", {"class": "table-reponsive"})
    
    if not table:
        print("Exchange rate table not found in HTML")
        return None
    
    data = []
    rows = table.find_all("tr")
    for row in rows[1:]:
        cells = row.find_all("td")
        if len(cells) >= 5:
            currency_code = cells[0].find_all("span", {"class": "mobile-content"})[-1].text.strip()
            
            cash = cells[2].find_all("span", {"class": "mobile-content"})[-1].text.strip()
            cash = None if cash == "-" else cash
            
            transfer = cells[3].find_all("span", {"class": "mobile-content"})[-1].text.strip()
            transfer = None if transfer == "-" else transfer
            
            sell = cells[4].find_all("span", {"class": "mobile-content"})[-1].text.strip()
            sell = None if sell == "-" else sell
            
            data.append({
                "Date": date_str,
                "CurrencyCode": currency_code,
                "Cash": cash,
                "Transfer": transfer,
                "Sell": sell
            })
    
    return data


def crawl_exchange_rates_bidv(
        start_date: datetime,
        end_date: datetime,
        file_path: str
    ):
    """Crawl exchange rates for dates from start_date to end_date with interval_months."""
    print(f"Starting crawl from {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}")

    driver = setup_driver()
    all_data = []
    
    try:
        driver.get("https://bidv.com.vn/en/ty-gia-ngoai-te")
        time.sleep(3)

        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%d/%m/%Y')

            if set_date_and_search_bidv(driver, date_str):
                html_content = driver.page_source
                data = extract_exchange_rates_bidv(html_content, date_str)
                
                if data:
                    all_data.extend(data)
                    print(f"Extracted {len(data)} currency rates for {date_str}")
                else:
                    print(f"No data extracted for {date_str}")

            current_date += timedelta(days=1)
            time.sleep(3)
        
    except Exception as e:
        print(f"Error during crawling: {e}")
    finally:
        driver.quit()
    
    if all_data:
        save_to_csv(all_data, file_path)
    else:
        print("No data was collected.")
