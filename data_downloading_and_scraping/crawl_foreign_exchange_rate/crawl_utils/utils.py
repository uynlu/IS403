import pandas as pd
import json
import os

from selenium import webdriver
from selenium.webdriver.chrome.service import Service


webdriver_path = "./chromedriver-win64/chromedriver.exe"

def setup_driver():
    """Set up and return a Chrome WebDriver."""
    service = Service(executable_path=webdriver_path)

    options = webdriver.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(service=service, options=options)

    return driver


def save_to_csv(
        data: list,
        file_path: str
    ):
    """Save data to a CSV file."""
    if not isinstance(data, pd.DataFrame):
        if not data:
            print("No data to save.")
            return
        data = pd.DataFrame(data)

    data.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")


def save_to_json(
        data: dict,
        file_path: str
    ):
    """Save data to a JSON file."""
    if not data:
        print("No data to save.")
        return
    
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def merge_csv(
        bank_stock: str,
        saved_path: str
    ):
    bank_31122021 = pd.read_csv(f"./dataset/raw_dataset/foreign_exchange_rate/{bank_stock}/{bank_stock}_foreign_exchange_rate_31122021.csv")

    bank_01012022_31032022 = pd.read_csv(f"./dataset/raw_dataset/foreign_exchange_rate/{bank_stock}/{bank_stock}_foreign_exchange_rate_01012022_31032022.csv")
    bank_01042022_30062022 = pd.read_csv(f"./dataset/raw_dataset/foreign_exchange_rate/{bank_stock}/{bank_stock}_foreign_exchange_rate_01042022_30062022.csv")
    bank_01072022_30092022 = pd.read_csv(f"./dataset/raw_dataset/foreign_exchange_rate/{bank_stock}/{bank_stock}_foreign_exchange_rate_01072022_30092022.csv")
    bank_01102022_31122022 = pd.read_csv(f"./dataset/raw_dataset/foreign_exchange_rate/{bank_stock}/{bank_stock}_foreign_exchange_rate_01102022_31122022.csv")
    
    bank_01012023_31032023 = pd.read_csv(f"./dataset/raw_dataset/foreign_exchange_rate/{bank_stock}/{bank_stock}_foreign_exchange_rate_01012023_31032023.csv")
    bank_01042023_30062023 = pd.read_csv(f"./dataset/raw_dataset/foreign_exchange_rate/{bank_stock}/{bank_stock}_foreign_exchange_rate_01042023_30062023.csv")
    bank_01072023_30092023 = pd.read_csv(f"./dataset/raw_dataset/foreign_exchange_rate/{bank_stock}/{bank_stock}_foreign_exchange_rate_01072023_30092023.csv")
    bank_01102023_31122023 = pd.read_csv(f"./dataset/raw_dataset/foreign_exchange_rate/{bank_stock}/{bank_stock}_foreign_exchange_rate_01102023_31122023.csv")
    
    bank_01012024_31032024 = pd.read_csv(f"./dataset/raw_dataset/foreign_exchange_rate/{bank_stock}/{bank_stock}_foreign_exchange_rate_01012024_31032024.csv")
    bank_01042024_30062024 = pd.read_csv(f"./dataset/raw_dataset/foreign_exchange_rate/{bank_stock}/{bank_stock}_foreign_exchange_rate_01042024_30062024.csv")
    bank_01072024_30092024 = pd.read_csv(f"./dataset/raw_dataset/foreign_exchange_rate/{bank_stock}/{bank_stock}_foreign_exchange_rate_01072024_30092024.csv")
    bank_01102024_31122024 = pd.read_csv(f"./dataset/raw_dataset/foreign_exchange_rate/{bank_stock}/{bank_stock}_foreign_exchange_rate_01102024_31122024.csv")

    bank_01012025_31032025 = pd.read_csv(f"./dataset/raw_dataset/foreign_exchange_rate/{bank_stock}/{bank_stock}_foreign_exchange_rate_01012025_31032025.csv")

    bank_01012022_31032025 = pd.concat([
        bank_31122021,

        bank_01012022_31032022,
        bank_01042022_30062022,
        bank_01072022_30092022,
        bank_01102022_31122022,
        
        bank_01012023_31032023,
        bank_01042023_30062023,
        bank_01072023_30092023,
        bank_01102023_31122023,

        bank_01012024_31032024,
        bank_01042024_30062024,
        bank_01072024_30092024,
        bank_01102024_31122024,

        bank_01012025_31032025
    ])

    save_to_csv(bank_01012022_31032025, os.path.join(saved_path, f"{bank_stock}_foreign_exchange_rate.csv")) 
    