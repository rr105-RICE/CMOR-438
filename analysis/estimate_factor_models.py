from __future__ import annotations

import io
import math
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import requests


TICKERS: Sequence[str] = (
    "NVDA",
    "AAPL",
    "MSFT",
    "AMZN",
    "TSLA",
    "JPM",
    "ORCL",
    "XOM",
    "NFLX",
    "PLTR",
)

MARKET_TICKER = "SPX"
LOOKBACK_YEARS = 5


class DataUnavailableError(RuntimeError):
    """Raised when external datasets cannot be retrieved."""


@dataclass(frozen=True)
class RegressionResult:
    ticker: str
    model: str
    variable: str
    estimate: float
    t_stat: float | None
    significance: str
    r_squared: float | None
    observations: int


def _stooq_symbol(ticker: str) -> str:
    if ticker.upper() == "SPX":
        return "^spx"
    return f"{ticker.lower()}.us"


def fetch_stooq_daily_returns(ticker: str, start_date: datetime) -> pd.DataFrame:
    symbol = _stooq_symbol(ticker)
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"

    df = pd.read_csv(url, parse_dates=["Date"])
    if df.empty:
        raise DataUnavailableError(f"Stooq returned no data for {ticker}")

    df = df[df["Date"] >= start_date].sort_values("Date")
    df["Return"] = df["Close"].pct_change()
    return df[["Date", "Return"]].dropna().rename(columns={"Return": f"{ticker.upper()}_RET"})


def fetch_fama_french_daily_factors(start_date: datetime) -> pd.DataFrame:
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    response = requests.get(url, timeout=60)
    if response.status_code != 200:
        raise DataUnavailableError("Unable to download Fama-French factors")

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        name = zf.namelist()[0]
        raw = zf.read(name).decode("latin1")

    lines = raw.splitlines()
    try:
        header_idx = next(i for i, line in enumerate(lines) if line.startswith(",Mkt-RF"))
    except StopIteration as exc:
        raise DataUnavailableError("Unexpected Fama-French file format") from exc

    data_lines = []
    for line in lines[header_idx + 1 :]:
        if not line.strip():
            break
        data_lines.append(line)

    csv_payload = "Date" + lines[header_idx] + "\n" + "\n".join(data_lines)
    df = pd.read_csv(io.StringIO(csv_payload))

    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df = df[df["Date"] >= start_date].sort_values("Date").reset_index(drop=True)

    for column in ("Mkt-RF", "SMB", "HML", "RF"):
        df[column] = df[column] / 100.0

    return df


def regress(y: np.ndarray, x: np.ndarray) -> Dict[str, np.ndarray]:
    if y.ndim != 1:
        raise ValueError("Endogenous vector must be one-dimensional")
    if x.ndim != 2:
        raise ValueError("Design matrix must be two-dimensional")
    if y.shape[0] != x.shape[0]:
        raise ValueError("The number of observations in y and X must match")

    n_obs, n_params = x.shape
    if n_obs <= n_params:
        raise ValueError("Not enough observations to estimate the model")

    xtx = x.T @ x
    xty = x.T @ y
    beta = np.linalg.solve(xtx, xty)
    fitted = x @ beta
    residuals = y - fitted

    dof = n_obs - n_params
    sigma2 = residuals @ residuals / dof
    cov_beta = sigma2 * np.linalg.inv(xtx)

    stderr = np.sqrt(np.diag(cov_beta))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.where(stderr > 0, beta / stderr, np.nan)

    ss_tot = ((y - y.mean()) ** 2).sum()
    r_squared = 1.0 - (residuals @ residuals) / ss_tot if ss_tot > 0 else np.nan

    return {
        "beta": beta,
        "stderr": stderr,
        "t_stats": t_stats,
        "r_squared": r_squared,
        "n_obs": n_obs,
    }


def significance_stars(t_stat: float | None) -> str:
    if t_stat is None or math.isnan(t_stat):
        return ""

    absolute = abs(t_stat)
    if absolute >= 2.576:
        return "***"
    if absolute >= 1.96:
        return "**"
    if absolute >= 1.645:
        return "*"
    return ""


def estimate_models_for_ticker(
    ticker: str,
    stock_returns: pd.DataFrame,
    market_returns: pd.DataFrame,
    factors: pd.DataFrame,
) -> List[RegressionResult]:
    dfs = (
        stock_returns,
        market_returns,
        factors[["Date", "RF", "SMB", "HML"]],
    )
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="Date", how="inner")

    rf = merged["RF"].to_numpy()
    stock_ret = merged[f"{ticker}_RET"].to_numpy()
    market_ret = merged[f"{MARKET_TICKER}_RET"].to_numpy()
    smb = merged["SMB"].to_numpy()
    hml = merged["HML"].to_numpy()

    excess_stock = stock_ret - rf
    excess_market = market_ret - rf

    capm_x = np.column_stack([np.ones_like(excess_stock), excess_market])
    capm_res = regress(excess_stock, capm_x)
    capm_labels = ("Alpha", "Beta_Market")

    ff3_x = np.column_stack([np.ones_like(excess_stock), excess_market, smb, hml])
    ff3_res = regress(excess_stock, ff3_x)
    ff3_labels = ("Alpha", "Beta_Market", "Beta_SMB", "Beta_HML")

    results: List[RegressionResult] = []

    for label, estimate, t_stat in zip(capm_labels, capm_res["beta"], capm_res["t_stats"]):
        results.append(
            RegressionResult(
                ticker=ticker,
                model="CAPM",
                variable=label,
                estimate=float(estimate),
                t_stat=float(t_stat),
                significance=significance_stars(float(t_stat)),
                r_squared=capm_res["r_squared"],
                observations=capm_res["n_obs"],
            )
        )

    results.append(
        RegressionResult(
            ticker=ticker,
            model="CAPM",
            variable="R^2",
            estimate=float(capm_res["r_squared"]),
            t_stat=None,
            significance="",
            r_squared=capm_res["r_squared"],
            observations=capm_res["n_obs"],
        )
    )

    for label, estimate, t_stat in zip(ff3_labels, ff3_res["beta"], ff3_res["t_stats"]):
        results.append(
            RegressionResult(
                ticker=ticker,
                model="Fama-French 3",
                variable=label,
                estimate=float(estimate),
                t_stat=float(t_stat),
                significance=significance_stars(float(t_stat)),
                r_squared=ff3_res["r_squared"],
                observations=ff3_res["n_obs"],
            )
        )

    results.append(
        RegressionResult(
            ticker=ticker,
            model="Fama-French 3",
            variable="R^2",
            estimate=float(ff3_res["r_squared"]),
            t_stat=None,
            significance="",
            r_squared=ff3_res["r_squared"],
            observations=ff3_res["n_obs"],
        )
    )

    return results


def compile_results(results: Iterable[RegressionResult]) -> pd.DataFrame:
    records = []
    for item in results:
        records.append(
            {
                "Ticker": item.ticker,
                "Model": item.model,
                "Variable": item.variable,
                "Estimate": item.estimate,
                "t_stat": item.t_stat,
                "Significance": item.significance,
                "R^2": item.r_squared,
                "Obs": item.observations,
            }
        )
    df = pd.DataFrame(records)
    df.sort_values(["Model", "Ticker", "Variable"], inplace=True)
    return df


def format_table(df: pd.DataFrame) -> str:
    display_df = df.copy()
    display_df["Estimate"] = display_df["Estimate"].map(lambda x: f"{x:.4f}")
    display_df["t_stat"] = display_df["t_stat"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    display_df["R^2"] = display_df.apply(
        lambda row: f"{row['R^2']:.3f}" if row["Variable"] == "R^2" else "",
        axis=1,
    )
    display_df["Obs"] = display_df.apply(
        lambda row: f"{int(row['Obs'])}" if row["Variable"] == "Alpha" else "",
        axis=1,
    )
    display_df = display_df[["Model", "Ticker", "Variable", "Estimate", "t_stat", "Significance", "R^2", "Obs"]]

    headers = list(display_df.columns)
    separator = ["---"] * len(headers)
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(separator) + " |"]

    for _, row in display_df.iterrows():
        cells = [str(row[col]) if row[col] != "" else "" for col in headers]
        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join(rows)


def run_estimation(
    end_date: date | None = None,
    lookback_years: int = LOOKBACK_YEARS,
) -> pd.DataFrame:
    if end_date is None:
        end_date = datetime.utcnow().date()
    start_date = pd.Timestamp(end_date - timedelta(days=lookback_years * 365))

    stock_returns: Dict[str, pd.DataFrame] = {}
    for ticker in TICKERS:
        stock_returns[ticker] = fetch_stooq_daily_returns(ticker, start_date)

    market_returns = fetch_stooq_daily_returns(MARKET_TICKER, start_date)
    factors = fetch_fama_french_daily_factors(start_date)

    all_results: List[RegressionResult] = []
    for ticker in TICKERS:
        ticker_results = estimate_models_for_ticker(
            ticker,
            stock_returns[ticker],
            market_returns,
            factors,
        )
        all_results.extend(ticker_results)

    return compile_results(all_results)


def main() -> None:
    df = run_estimation()
    print(format_table(df))


if __name__ == "__main__":
    main()
