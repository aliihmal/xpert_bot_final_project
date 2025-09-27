
import datetime
import logging
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# ----------------------------
# Configuration (change here)
# ----------------------------
TICKERS = ["BAC", "JPM", "GS", "C", "WFC", "MS"]
START_DATE = "2006-01-01"   # can be datetime.date objects as well
END_DATE = "2016-01-01"
ANNUALIZATION_FACTOR = np.sqrt(252)  # used to annualize rolling volatility
ROLLING_WINDOW = 30  # days for rolling volatility
RISK_FREE_RATE = 0.0  # annual risk free rate for Sharpe (0 here for simplicity)

# Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ----------------------------
# Helper functions
# ----------------------------
def download_tickers(tickers: List[str], start: str, end: str) -> pd.DataFrame:

    try:
        logging.info(f"Downloading data for {tickers} from {start} to {end} ...")
        # Use yf.download on the list to get a multi-ticker dataframe directly (more memory-friendly)
        raw = yf.download(tickers, start=start, end=end, group_by='ticker', threads=True, progress=False)
        # yfinance sometimes returns a single-level DataFrame if only one ticker; handle that
        if isinstance(raw.columns, pd.MultiIndex) and raw.columns.nlevels == 2:
            # raw currently has (column, ticker) ordering; reshape to (ticker, column)
            # Reorganize to (Ticker, Info)
            # Some yfinance returns columns as (pricefield, ticker); fix to (ticker, pricefield)
            # Build standardized multiindex
            frames = []
            for t in tickers:
                if t in raw.columns.get_level_values(1):
                    df_t = raw.xs(t, axis=1, level=1).copy()
                elif t in raw.columns:
                    df_t = raw[t].copy()
                else:
                    df_t = None
                if df_t is not None:
                    # Ensure consistent column names: ['Open','High','Low','Close','Adj Close','Volume']
                    # Keep as is and add prefix key
                    df_t.columns = pd.Index(df_t.columns)
                    # create a multiindex by concatenating
                    df_t.columns = pd.MultiIndex.from_product([[t], df_t.columns])
                    frames.append(df_t)
            if not frames:
                raise ValueError("Downloaded data is empty or formatted unexpectedly.")
            combined = pd.concat(frames, axis=1)
        else:
            # If yfinance returned normal multiindex as (Ticker, Field) already
            # or single ticker
            if isinstance(raw.columns, pd.MultiIndex):
                combined = raw.copy()
            else:
                # single ticker (or weird format). Ensure MultiIndex columns (Ticker, Field)
                # attempt to detect the ticker from tickers[0]
                df_single = raw.copy()
                df_single.columns = pd.MultiIndex.from_product([[tickers[0]], df_single.columns])
                combined = df_single

        # Normalize column names: set names for multiindex
        combined.columns.names = ['Ticker', 'Info']
        logging.info("Download finished.")
        return combined

    except Exception as e:
        logging.error("Failed to download data: %s", e)
        raise


def compute_daily_returns(stock_df: pd.DataFrame) -> pd.DataFrame:

    closes = stock_df.xs('Close', axis=1, level='Info')
    returns = closes.pct_change().add_prefix('').rename(columns=lambda c: f"{c} Return")
    return returns


def compute_rolling_volatility(returns_df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:

    rolling_vol = returns_df.rolling(window=window).std() * ANNUALIZATION_FACTOR
    return rolling_vol


def compute_overnight_gap(stock_df: pd.DataFrame) -> pd.DataFrame:

    opens = stock_df.xs('Open', axis=1, level='Info')
    prev_closes = stock_df.xs('Close', axis=1, level='Info').shift(1)
    overnight_gap = (opens - prev_closes) / prev_closes * 100.0
    return overnight_gap


def compute_sharpe_ratio(returns_df: pd.DataFrame, annual_risk_free: float = RISK_FREE_RATE) -> pd.Series:

    rf_daily = (1 + annual_risk_free) ** (1 / 252) - 1
    mean_daily = returns_df.mean()
    std_daily = returns_df.std()
    sharpe = (mean_daily - rf_daily) / std_daily * np.sqrt(252)
    sharpe.name = "Sharpe"
    return sharpe


def compute_max_drawdown(price_df: pd.DataFrame) -> pd.Series:

    drawdowns = {}
    for col in price_df.columns:
        series = price_df[col].dropna()
        running_max = series.cummax()
        drawdown = (series - running_max) / running_max
        drawdowns[col] = drawdown.min()
    return pd.Series(drawdowns)


def plot_price_series(price_df: pd.DataFrame, title: str = "Close Prices"):
    plt.figure(figsize=(12, 5))
    for col in price_df.columns:
        plt.plot(price_df.index, price_df[col], label=col)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main(tickers: List[str], start: str, end: str):
    # 1. Download
    stock_df = download_tickers(tickers=tickers, start=start, end=end)

    # 2. Basic checks & memory optimization: keep only what we need going forward
    # We'll keep 'Open', 'Close', and 'Volume' explicitly (others can be added when needed)
    required_info = ['Open', 'Close', 'Volume', 'Adj Close']
    existing_info = stock_df.columns.get_level_values('Info').unique().tolist()
    infos_to_keep = [i for i in required_info if i in existing_info]
    # build new df that keeps only selected info
    keep_cols = [c for c in stock_df.columns if c[1] in infos_to_keep]
    stock_df = stock_df.loc[:, keep_cols]

    # 3. Compute returns
    returns = compute_daily_returns(stock_df)

    # 4. Quick statistical summary
    print("=== Basic return statistics ===")
    print(returns.describe().loc[['mean', 'std', 'min', 'max']])

    # 5. Pairplot (sample to avoid extremely heavy plotting)
    try:
        sample_for_pairplot = returns.dropna().sample(min(400, len(returns.dropna())), random_state=1)
        sns.pairplot(sample_for_pairplot)
        plt.suptitle("Pairwise relationships of returns (sample)", y=1.02)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.warning("Pairplot failed or was skipped due to size/format: %s", e)

    # 6. Plot prices
    closes = stock_df.xs('Close', axis=1, level='Info')
    plot_price_series(closes, title=f"Close Prices ({start} to {end})")

    # 7. Correlation heatmap of close prices
    corr_df = closes.corr()
    plt.figure(figsize=(7, 5))
    sns.heatmap(corr_df, annot=True, fmt=".2f")
    plt.title("Correlation Matrix of Close Prices")
    plt.tight_layout()
    plt.show()

    # 8. Rolling (30-day) annualized volatility
    rolling_vol = compute_rolling_volatility(returns)
    print("\nMax rolling volatility dates (per ticker):")
    # show the date when volatility peaked for each ticker
    for col in returns.columns:
        idx = rolling_vol[col].idxmax()
        print(col, idx)

    # 9. Sharpe ratio and max drawdown (bonus risk metrics)
    sharpe = compute_sharpe_ratio(returns)
    print("\n=== Annualized Sharpe Ratio ===")
    print(sharpe.sort_values(ascending=False))

    max_dd = compute_max_drawdown(closes)
    print("\n=== Max Drawdown (lowest negative value) ===")
    print(max_dd.sort_values())

    # 10. Overnight gap analysis
    overnight_gap = compute_overnight_gap(stock_df)
    print("\nOvernight Gap (sample stats):")
    print(overnight_gap.describe().T)

    # Example: count large negative gaps (< -3%)
    large_negative = (overnight_gap < -3).sum()
    print("\nDays with overnight gap < -3%:")
    print(large_negative)

    # Clean up large temporary variables (help memory)
    del stock_df
    # return results for further use if desired
    return {
        "returns": returns,
        "rolling_volatility": rolling_vol,
        "overnight_gap": overnight_gap,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


if __name__ == "__main__":
    results = main(TICKERS, START_DATE, END_DATE)