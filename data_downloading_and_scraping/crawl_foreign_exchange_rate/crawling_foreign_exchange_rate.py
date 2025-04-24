from dataset_preparation.crawl_foreign_exchange_rate.crawl_utils import (
    crawl_exchange_rates_bidv,
    crawl_exchange_rates_vcb,
    crawl_exchange_rates_tcb
)
from dataset_preparation.crawl_foreign_exchange_rate.crawl_utils.utils import merge_csv

import argparse
from datetime import datetime


parser = argparse.ArgumentParser(description="Crawl foreign exchange rate")
parser.add_argument("--merge", type=bool, default=False)
parser.add_argument("--bank", type=str, required=True)
parser.add_argument("--start_date", type=str)
parser.add_argument("--end_date", type=str)
parser.add_argument("--saved_path", type=str)
args = parser.parse_args()

if args.merge:
    if args.bank == "BIDV":
        merge_csv("bid", "./dataset/raw_dataset/foreign_exchange_rate")
    elif args.bank == "Vietcombank":
        merge_csv("vcb", "./dataset/raw_dataset/foreign_exchange_rate")
    elif args.bank == "Techcombank":
        merge_csv("tcb", "./dataset/raw_dataset/foreign_exchange_rate")

else:
    start_date = datetime(*(map(int, args.start_date.split("-"))))
    end_date = datetime(*(map(int, args.end_date.split("-"))))
    if args.bank == "BIDV":
        crawl_exchange_rates_bidv(start_date, end_date, args.saved_path)
    elif args.bank == "Vietcombank":
        crawl_exchange_rates_vcb(start_date, end_date, args.saved_path)
    elif args.bank == "Techcombank":
        crawl_exchange_rates_tcb(start_date, end_date, args.saved_path)
