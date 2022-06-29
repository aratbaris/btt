"""Financial data sources."""

import requests
import pandas as pd


def get_stock_data_from_finmodelprep(api_key, stock, days=1200):
    """
    Get historical price data.

    Parameters
    ----------
    api_key : str
        Financial Modeling Prep API Key.
    stock : str
        Stock symbol.
    days : int, optional
        Total number of days from recent to historical. The default is 1200.

    Returns
    -------
    stockprices : pandas.DataFrame
        Listing date and close price.

    """
    stockprices = requests.get(
        'https://financialmodelingprep.com/api/v3/historical-price-full/' +
        f'{stock}?serietype=line&apikey={api_key}').json()
    stockprices = pd.DataFrame.from_dict(stockprices['historical'][0:days])
    stockprices = stockprices.set_index('date')
    # reverse dates in the index to have more recent days at the end
    stockprices = stockprices.iloc[::-1]

    return stockprices
