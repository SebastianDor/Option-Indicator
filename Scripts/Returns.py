from datetime import datetime
import yfinance as yf
import numpy as np
import pandas as pd


def get_index_returns(
    tickers: list[str] = ["^STOXX", "^STOXX50E", "^AEX"],
    years_back: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date.replace(year=end_date.year - years_back)

    df_multi = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False
    )

    returns = pd.DataFrame()
    cum_returns = pd.DataFrame()

    for ticker in tickers:
        prices = df_multi[ticker]["Close"]
        log_ret = np.log(prices / prices.shift(1))
        cum_ret = np.exp(log_ret.cumsum()) - 1

        returns[ticker] = log_ret
        cum_returns[ticker] = cum_ret

    return returns, cum_returns
