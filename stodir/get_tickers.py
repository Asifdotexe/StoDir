import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor

from stodir.config import PORTFOLIOS

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)


def scrape_and_save_tickers(portfolio_config: dict):
    """Scrapes the list of portfolio tickers for the provided portfolio from Wikipedia."""
    name = portfolio_config["name"]
    url = portfolio_config["url"]
    ticker_col = portfolio_config["ticker_column_index"]
    suffix = portfolio_config.get("ticker_suffix", "")

    logging.info(f"Fetching tickers for {name.upper()}...")

    HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)\
                              AppleWebKit/537.36 (KHTML, like Gecko)\
                              Chrome/94.0.4606.81 Safari/537.36"}
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    table = soup.find("table", {"id": "constituents"})
    if not table:
        raise ValueError(f"Could not find the constituents table on the page for {name}.")

    tickers = []
    for row in table.find_all("tr")[1:]:
        cells = row.find_all("td")
        if len(cells) > ticker_col:
            ticker = cells[ticker_col].text.strip()
            if "." in ticker:
                ticker = ticker.replace('.', '-')
            tickers.append(f"{ticker}{suffix}")

    output_path = os.path.join("data", f"{name}_tickers.txt")
    with open(output_path, "w") as f:
        f.write("\n".join(tickers))

    logging.info(f"Successfully saved {len(tickers)} tickers for {name.upper()} to {output_path}")


def main() -> None:

    os.makedirs("data", exist_ok=True)
    logging.info("--- Starting Ticker Acquisition ---")

    with ThreadPoolExecutor() as executor:
        executor.map(scrape_and_save_tickers, PORTFOLIOS)

    logging.info("--- Ticker Acquisition Complete ---")


if __name__ == "__main__":
    main()