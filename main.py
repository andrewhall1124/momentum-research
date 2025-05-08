import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium", app_title="Momentum Research")


@app.cell
def _():
    import polars as pl
    import polars_ds as pds
    import marimo as mo
    import datetime as dt
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from great_tables import GT, md, html
    return GT, dt, mo, np, pl, plt, sns


@app.cell
def _(dt, mo):
    start_date = mo.ui.date(label="Start Date", value=dt.date(1995, 1, 1))  # 1965-1-1
    end_date = mo.ui.date(label="End Date", value=dt.date(2024, 12, 31))  # 1989-12-31
    window = mo.ui.number(label="Feature Window", value=230)
    skip = mo.ui.number(label="Feature Skip", value=22)

    n_bins = mo.ui.number(start=2, stop=20, value=10, label="Number of Bins")

    mo.vstack(items=[start_date, end_date, window, skip, n_bins])
    return end_date, n_bins, skip, start_date, window


@app.cell
def _(pl):
    data = pl.read_parquet('data/data.parquet')
    return (data,)


@app.cell
def _(data, end_date, pl, skip, start_date, window):
    features = (
        data
        .filter(
            pl.col('date').is_between(start_date.value, end_date.value)
        )
        .with_columns(
            pl.col('ret')
            .log1p()
            .rolling_sum(window_size=window.value)
            .shift(skip.value)
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
def _(labels, pl, portfolios):
    p_labels = [f'P{i}' for i in range(10)]
    label_mapping = {label: p_label for label, p_label in zip(labels, p_labels)}

    summary = (
        portfolios
        .unpivot(index='date', variable_name='portfolio', value_name='ret')
        .group_by('portfolio')
        .agg(
            pl.col('ret').mul(100).mean().alias('mean'),
            pl.col('ret').mul(100).std().alias('stdev'),
            # pds.ttest_1samp('ret', pop_mean=0, alternative='two-sided').alias('stats')
        )
        .with_columns(
            pl.col('mean').mul(252),
            pl.col('stdev').mul(pl.lit(252).sqrt())
        )
        .with_columns(
            pl.col('mean').truediv(pl.col('stdev')).alias('sharpe')
        )
        .sort('portfolio')
    )

    table = (
        summary
        .drop('portfolio')
        .rename({
            'mean': 'Mean (%)',
            'stdev': 'Std. Dev. (%)',
            'sharpe': 'Sharpe'
        })
        .transpose(include_header=True, column_names=[*labels, 'Spread'], header_name='Portfolio')
        .rename(label_mapping)
    )
    return p_labels, summary, table


@app.cell
def _(GT, p_labels, summary, table):
    max_return = summary['mean'].max()
    min_return = summary['mean'].min()
    max_sharpe = summary['sharpe'].max()
    min_sharpe = summary['sharpe'].min()

    gt_tbl = (
        GT(table)
        .tab_header(title='Portfolios Summary (Annualized)')
        .fmt_number(columns=[*p_labels, 'Spread'], decimals=2)
        .data_color(
            columns=[*p_labels],
            palette=['red', 'white', 'green'],
            rows=0,
            domain=[min_return, max_return]
        )
        .data_color(
            columns=[*p_labels],
            palette=['red', 'white', 'green'],
            rows=2,
            domain=[min_sharpe, max_sharpe]
        )
    )

    gt_tbl
    return


@app.cell
def _(cummulative_returns, labels, n_bins, np, plt, portfolios, sns):
    sharpe = portfolios['spread'].mean() / portfolios['spread'].std() * np.sqrt(252)

    plt.figure(figsize=(10, 6))

    colors = sns.color_palette(palette="coolwarm", n_colors=n_bins.value)

    for label, color in zip(labels, colors):
        sns.lineplot(cummulative_returns, x='date', y=label, color=color, label=label)

    sns.lineplot(cummulative_returns, x='date', y='spread', color='green', label='Spread')

    plt.title(f"Backtest ({sharpe:.2f})")

    plt.xlabel(None)
    plt.ylabel("Cummulative Sum Returns (%)")

    plt.legend(title="Portfolio", loc="upper left", bbox_to_anchor=(1, 1))
    return


if __name__ == "__main__":
    app.run()
