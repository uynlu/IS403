from data_downloading_and_scraping.crawl_foreign_exchange_rate.crawl_utils.utils import save_to_csv

import requests

from datetime import datetime, timedelta


def crawl_exchange_rates_vcb(
        start_date: datetime,
        end_date: datetime,
        file_path: str
    ):
    print(f"Starting crawl from {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}")

    all_data = []

    current_date = start_date
    while current_date <= end_date:
        url = f"https://www.vietcombank.com.vn/api/exchangerates?date={current_date.year}-{current_date.month}-{current_date.day}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data:
                for i in range(20):
                    all_data.append({
                        "Date": current_date.strftime('%d/%m/%Y'),
                        "CurrencyCode": data["Data"][i]["currencyCode"],
                        "Cash": data["Data"][i]["cash"],
                        "Transfer": data["Data"][i]["transfer"],
                        "Sell": data["Data"][i]["sell"]
                    })
                print(f"Extracted {data['Count']} currency rates for {current_date.strftime('%d/%m/%Y')}")
            else:
                print(f"No data extracted for {current_date.strftime('%d/%m/%Y')}")
        current_date += timedelta(days=1)

    if all_data:
        save_to_csv(all_data, file_path)
    else:
        print("No data was collected.")
