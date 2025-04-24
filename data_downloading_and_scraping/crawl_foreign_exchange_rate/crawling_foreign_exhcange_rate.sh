#!/bin/sh
$env:PYTHONPATH="D:\UIT\HK\HK6\IS403\Đồ án\Source code"


# ========== BIDV =============
# --- 2021 ---
python dataset_preparation/crawl_foreign_exchange_rate/crawling_foreign_exchange_rate.py --bank BIDV --start_date 2021-12-31 --end_date 2021-12-31 --saved_path dataset/raw_dataset/foreign_exchange_rate/bid/bid_foreign_exchange_rate_31122021.csv
# --- 2022 ---
python dataset_preparation/crawl_foreign_exchange_rate/crawling_foreign_exchange_rate.py --bank BIDV --start_date 2022-01-01 --end_date 2022-03-31 --saved_path dataset/raw_dataset/foreign_exchange_rate/bid/bid_foreign_exchange_rate_01012022_31032022.csv
python dataset_preparation/crawl_foreign_exchange_rate/crawling_foreign_exchange_rate.py --bank BIDV --start_date 2022-04-01 --end_date 2022-06-30 --saved_path dataset/raw_dataset/foreign_exchange_rate/bid/bid_foreign_exchange_rate_01042022_30062022.csv
python dataset_preparation/crawl_foreign_exchange_rate/crawling_foreign_exchange_rate.py --bank BIDV --start_date 2022-07-01 --end_date 2022-09-30 --saved_path dataset/raw_dataset/foreign_exchange_rate/bid/bid_foreign_exchange_rate_01072022_30092022.csv
python dataset_preparation/crawl_foreign_exchange_rate/crawling_foreign_exchange_rate.py --bank BIDV --start_date 2022-10-01 --end_date 2022-12-31 --saved_path dataset/raw_dataset/foreign_exchange_rate/bid/bid_foreign_exchange_rate_01102022_31122022.csv
# --- 2023 ---
python dataset_preparation/crawl_foreign_exchange_rate/crawling_foreign_exchange_rate.py --bank BIDV --start_date 2023-01-01 --end_date 2023-03-31 --saved_path dataset/raw_dataset/foreign_exchange_rate/bid/bid_foreign_exchange_rate_01012023_31032023.csv
python dataset_preparation/crawl_foreign_exchange_rate/crawling_foreign_exchange_rate.py --bank BIDV --start_date 2023-04-01 --end_date 2023-06-30 --saved_path dataset/raw_dataset/foreign_exchange_rate/bid/bid_foreign_exchange_rate_01042023_30062023.csv
python dataset_preparation/crawl_foreign_exchange_rate/crawling_foreign_exchange_rate.py --bank BIDV --start_date 2023-07-01 --end_date 2023-09-30 --saved_path dataset/raw_dataset/foreign_exchange_rate/bid/bid_foreign_exchange_rate_01072023_30092023.csv
python dataset_preparation/crawl_foreign_exchange_rate/crawling_foreign_exchange_rate.py --bank BIDV --start_date 2023-10-01 --end_date 2023-12-31 --saved_path dataset/raw_dataset/foreign_exchange_rate/bid/bid_foreign_exchange_rate_01102023_31122023.csv
# --- 2024 ---
python dataset_preparation/crawl_foreign_exchange_rate/crawling_foreign_exchange_rate.py --bank BIDV --start_date 2024-01-01 --end_date 2024-03-31 --saved_path dataset/raw_dataset/foreign_exchange_rate/bid/bid_foreign_exchange_rate_01012024_31032024.csv
python dataset_preparation/crawl_foreign_exchange_rate/crawling_foreign_exchange_rate.py --bank BIDV --start_date 2024-04-01 --end_date 2024-06-30 --saved_path dataset/raw_dataset/foreign_exchange_rate/bid/bid_foreign_exchange_rate_01042024_30062024.csv
python dataset_preparation/crawl_foreign_exchange_rate/crawling_foreign_exchange_rate.py --bank BIDV --start_date 2024-07-01 --end_date 2024-09-30 --saved_path dataset/raw_dataset/foreign_exchange_rate/bid/bid_foreign_exchange_rate_01072024_30092024.csv
python dataset_preparation/crawl_foreign_exchange_rate/crawling_foreign_exchange_rate.py --bank BIDV --start_date 2024-10-01 --end_date 2024-12-31 --saved_path dataset/raw_dataset/foreign_exchange_rate/bid/bid_foreign_exchange_rate_01102024_31122024.csv
# --- 2025 ---
python dataset_preparation/crawl_foreign_exchange_rate/crawling_foreign_exchange_rate.py --bank BIDV --start_date 2025-01-01 --end_date 2025-03-31 --saved_path dataset/raw_dataset/foreign_exchange_rate/bid/bid_foreign_exchange_rate_01012025_31032025.csv
# ------------
python dataset_preparation/crawl_foreign_exchange_rate/crawling_foreign_exchange_rate.py --merge True --bank BIDV

# ========== Vietcombank =============
python dataset_preparation/crawl_foreign_exchange_rate/crawling_foreign_exchange_rate.py --bank Vietcombank --start_date 2021-12-31 --end_date 2025-03-31 --saved_path dataset/raw_dataset/foreign_exchange_rate/vcb_foreign_exchange_rate_31122021_31032025.csv

# ========== Techcombank =============
# --- 2021 ---
python dataset_preparation/crawl_foreign_exchange_rate/crawling_foreign_exchange_rate.py --bank Techcombank --start_date 2021-12-31 --end_date 2021-12-31 --saved_path dataset/raw_dataset/foreign_exchange_rate/tcb/tcb_foreign_exchange_rate_31122021.csv
# --- 2022 ---
python dataset_preparation/crawl_foreign_exchange_rate/crawling_foreign_exchange_rate.py --bank Techcombank --start_date 2022-01-01 --end_date 2022-03-31 --saved_path dataset/raw_dataset/foreign_exchange_rate/tcb/tcb_foreign_exchange_rate_01012022_31032022.csv
python dataset_preparation/crawl_foreign_exchange_rate/crawling_foreign_exchange_rate.py --bank Techcombank --start_date 2022-04-01 --end_date 2022-06-30 --saved_path dataset/raw_dataset/foreign_exchange_rate/tcb/tcb_foreign_exchange_rate_01042022_30062022.csv
python dataset_preparation/crawl_foreign_exchange_rate/crawling_foreign_exchange_rate.py --bank Techcombank --start_date 2022-07-01 --end_date 2022-09-30 --saved_path dataset/raw_dataset/foreign_exchange_rate/tcb/tcb_foreign_exchange_rate_01072022_30092022.csv
python dataset_preparation/crawl_foreign_exchange_rate/crawling_foreign_exchange_rate.py --bank Techcombank --start_date 2022-10-01 --end_date 2022-12-31 --saved_path dataset/raw_dataset/foreign_exchange_rate/tcb/tcb_foreign_exchange_rate_01102022_31122022.csv
# --- 2023 ---
python dataset_preparation/crawl_foreign_exchange_rate/crawling_foreign_exchange_rate.py --bank Techcombank --start_date 2023-01-01 --end_date 2023-03-31 --saved_path dataset/raw_dataset/foreign_exchange_rate/tcb/tcb_foreign_exchange_rate_01012023_31032023.csv
python dataset_preparation/crawl_foreign_exchange_rate/crawling_foreign_exchange_rate.py --bank Techcombank --start_date 2023-04-01 --end_date 2023-06-30 --saved_path dataset/raw_dataset/foreign_exchange_rate/tcb/tcb_foreign_exchange_rate_01042023_30062023.csv
python dataset_preparation/crawl_foreign_exchange_rate/crawling_foreign_exchange_rate.py --bank Techcombank --start_date 2023-07-01 --end_date 2023-09-30 --saved_path dataset/raw_dataset/foreign_exchange_rate/tcb/tcb_foreign_exchange_rate_01072023_30092023.csv
python dataset_preparation/crawl_foreign_exchange_rate/crawling_foreign_exchange_rate.py --bank Techcombank --start_date 2023-10-01 --end_date 2023-12-31 --saved_path dataset/raw_dataset/foreign_exchange_rate/tcb/tcb_foreign_exchange_rate_01102023_31122023.csv
# --- 2024 ---
python dataset_preparation/crawl_foreign_exchange_rate/crawling_foreign_exchange_rate.py --bank Techcombank --start_date 2024-01-01 --end_date 2024-03-31 --saved_path dataset/raw_dataset/foreign_exchange_rate/tcb/tcb_foreign_exchange_rate_01012024_31032024.csv
python dataset_preparation/crawl_foreign_exchange_rate/crawling_foreign_exchange_rate.py --bank Techcombank --start_date 2024-04-01 --end_date 2024-06-30 --saved_path dataset/raw_dataset/foreign_exchange_rate/tcb/tcb_foreign_exchange_rate_01042024_30062024.csv
python dataset_preparation/crawl_foreign_exchange_rate/crawling_foreign_exchange_rate.py --bank Techcombank --start_date 2024-07-01 --end_date 2024-09-30 --saved_path dataset/raw_dataset/foreign_exchange_rate/tcb/tcb_foreign_exchange_rate_01072024_30092024.csv
python dataset_preparation/crawl_foreign_exchange_rate/crawling_foreign_exchange_rate.py --bank Techcombank --start_date 2024-10-01 --end_date 2024-12-31 --saved_path dataset/raw_dataset/foreign_exchange_rate/tcb/tcb_foreign_exchange_rate_01102024_31122024.csv
# --- 2025 ---
python dataset_preparation/crawl_foreign_exchange_rate/crawling_foreign_exchange_rate.py --bank Techcombank --start_date 2025-01-01 --end_date 2025-03-31 --saved_path dataset/raw_dataset/foreign_exchange_rate/tcb/tcb_foreign_exchange_rate_01012025_31032025.csv
# ------------
python dataset_preparation/crawl_foreign_exchange_rate/crawling_foreign_exchange_rate.py --merge True --bank Techcombank
