import marimo

__generated_with = "0.13.4"
app = marimo.App(width="medium", app_title="Momentum Research")


@app.cell
def _():
    import polars as pl
    import marimo as mo
    import datetime as dt
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    return dt, mo, np, pl, plt, sns


@app.cell
def _(dt, mo):
    start_date = mo.ui.date(label="Start Date", value=dt.date(1995, 1, 1))  # 1965-1-1
    end_date = mo.ui.date(label="End Date", value=dt.date(2024, 12, 31))  # 1989-12-31

    n_bins = mo.ui.number(start=2, stop=20, value=10, label="Number of Bins")

    mo.vstack(items=[start_date, end_date, n_bins])
    return end_date, n_bins, start_date


@app.cell
def _(end_date, pl, start_date):
    data = (
        pl.scan_parquet('data/data.parquet')
        .filter(
            pl.col('date').is_between(start_date.value, end_date.value)
        )
        .collect()
    )
    data.head(5)
    return (data,)


@app.cell
def _():
    window = 230
    skip = 22
    return skip, window


@app.cell
def _(data, pl, skip, window):
    features = (
        data.with_columns(
            pl.col('ret')
            .log1p()
            .rolling_sum(window_size=window)
            .shift(skip)
            .over('permno')
            .alias('momentum')
        )
        .with_columns(
            pl.col('prc').shift(1).over('permno').alias('prc_lag')
        )
        .filter(pl.col('prc_lag').gt(5))
        .drop_nulls('momentum')
        .sort(['permno', 'date'])
    )
    return (features,)


@app.cell
def _(features, n_bins, pl):
    labels = [str(i) for i in range(n_bins.value)]

    bins = (
        features.with_columns(
            pl.col('momentum')
            .qcut(n_bins.value, labels=labels)
            .over('date')
            .alias('bin')
        )
        .sort(['permno', 'date'])
    )
    return bins, labels


@app.cell
def _(bins, n_bins, pl):
    portfolios = (
        bins.group_by(['date', 'bin'])
        .agg(pl.col('ret').mean())
        .pivot(on='bin', index='date', values='ret')
        .with_columns(
            pl.col(str(n_bins.value - 1))
            .sub(pl.col('0'))
            .alias('spread')
        )
        .sort('date')
    )
    return (portfolios,)


@app.cell
def _(labels, pl, portfolios):
    cummulative_returns = (
        portfolios.with_columns(
            pl.col(*labels, 'spread')
            .log1p()
            .cum_sum()
            .mul(100)
        )
    )
    return (cummulative_returns,)


@app.cell
def _(cummulative_returns, labels, n_bins, np, plt, portfolios, sns):
    sharpe = portfolios['spread'].mean() / portfolios['spread'].std() * np.sqrt(252)

    plt.figure(figsize=(10, 6))

    colors = sns.color_palette(palette="coolwarm", n_colors=n_bins.value)

    for label, color in zip(labels, colors):
        sns.lineplot(cummulative_returns, x='date', y=label, color=color, label=label)

    sns.lineplot(cummulative_returns, x='date', y='spread', color='green', label='Spread')

    plt.title(f"Spread Portfolio Backtest ({sharpe:.2f})")

    plt.xlabel(None)
    plt.ylabel("Cummulative Sum Returns (%)")

    plt.legend(title="Portfolio", loc="upper left", bbox_to_anchor=(1, 1))
    return


if __name__ == "__main__":
    app.run()
