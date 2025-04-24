from dataset_preparation.crawl_foreign_exchange_rate.crawl_utils.utils import setup_driver, save_to_csv

import time
from datetime import datetime, timedelta

from selenium import webdriver
from selenium.webdriver.common.by import By

from bs4 import BeautifulSoup


def set_date_and_search_tcb(
        driver: webdriver.Chrome,
        date_str: str
    ):
    """Set the date in the input field and click the search button."""
    try:
        script = f"document.querySelector('.calendar__input-field').innerText = '{date_str}';"
        driver.execute_script(script)
        
        print(f"Successfully set date to {date_str} and searched")
        return True
    
    except Exception as e:
        print(f"Error setting date and searching: {e}")
        return False


def extract_exchange_rates_tcb(
        html_content: str,
        date_str: str
    ):
    """Extract exchange rates from the HTML content."""
    soup = BeautifulSoup(html_content, "html.parser")
    table = soup.find("div", {"class": "exchange-rate__table-content"})
    
    if not table:
        print("Exchange rate table not found in HTML")
        return None
    
    data = []
    rows = table.find("div", {"class": "exchange-rate-table-content"}).find_all("div", {"class": "exchange-rate__table-records"})
    for row in rows:
        cells = row.find_all("p")
        if len(cells) >= 5:
            currency_code = cells[0].text.strip()
            
            cash = cells[1].text.strip()

            transfer = cells[2].text.strip()
            
            sell = cells[4].text.strip()
            
            data.append({
                "Date": date_str,
                "CurrencyCode": currency_code,
                "Cash": cash,
                "Transfer": transfer,
                "Sell": sell
            })
    
    return data

def crawl_exchange_rates_tcb(
        start_date: datetime,
        end_date: datetime,
        file_path: str
    ):
    """Crawl exchange rates for dates from start_date to end_date with interval_months."""
    print(f"Starting crawl from {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}")

    driver = setup_driver()
    all_data = []
    
    try:
        driver.get("https://techcombank.com/cong-cu-tien-ich/ty-gia#ti-gia-hoi-doai")
        time.sleep(3)

        flag = False
        current_date = start_date
        while current_date <= (end_date + timedelta(days=1)):
            if flag == False:
                set_date_and_search_tcb(driver, current_date.strftime("%d/%m/%Y"))
                flag = True
            else:
                real_date_str = (current_date - timedelta(days=1)).strftime("%d/%m/%Y")
                date_str = current_date.strftime("%d/%m/%Y")

                null_data_element = driver.find_element(By.CLASS_NAME, "exchange-rate__empty-label")
                if set_date_and_search_tcb(driver, date_str):
                    if null_data_element.is_displayed() == False:
                        html_content = driver.page_source
                        data = extract_exchange_rates_tcb(html_content, real_date_str)
                        all_data.extend(data)
                        print(f"Extracted {len(data)} currency rates for {real_date_str}")
                    else:
                        print(f"No data extracted for {real_date_str}")

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
