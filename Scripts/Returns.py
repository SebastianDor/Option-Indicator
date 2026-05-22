import yfinance as yf

# Yahoo tickers for some major indices
symbols = [
    "^GSPC",     # S&P 500
    "^NDX",      # Nasdaq 100
    "^DJI",      # Dow Jones
    "^STOXX50E", # Euro Stoxx 50
    "^GDAXI",    # DAX
    "^FCHI",     # CAC 40
    "^FTSE"      # FTSE 100
]

data = yf.download(tickers=" ".join(symbols), period="1d", interval="1m")
print(data.tail())  # last few rows of 1-minute data (delayed)