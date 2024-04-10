""" zid_project2_etl.py

"""

# ----------------------------------------------------------------------------
# Part 4.1: import needed modules
# ----------------------------------------------------------------------------
# Create import statements to import all modules you need in this script
# Note: please keep the aliases consistent throughout the project.
#       For details, review the import statements in zid_project2_main.py

import numpy as np
import pandas as pd
import config as cfg
import util
import unittest


# ----------------------------------------------------------------------------
# Part 4.2: Complete the read_prc_csv function
# ----------------------------------------------------------------------------
def read_prc_csv(tic: str, start: str, end: str, prc_col='Adj Close') -> pd.Series:
    """ This function extracts and returns a Pandas Series of adjusted close prices
    for a specified ticker over a defined date range, sourced from a CSV file
    containing stock price information.

    Parameters
    ----------
    tic : str
        String with the ticker (can include lowercase and/or uppercase characters)

    start  :  str
        The inclusive start date for the date range of the output Pandas series
        For example: if you enter '2010-09-02', the function will include data from
        this date onwards. And make sure the provided start date is a valid
        calendar date.

    end  :  str
        The inclusive end date for the date range, which determines the final date
        included in the output Pandas series
        For example: if you enter '2010-12-20', the output series will encompass data
        up to and including December 20, 2010. And make sure the provided start date
        is a valid calendar date.

    prc_col: str, optional
        Column name of stock adjusted price in the imported CSV file

    Returns
    -------
    ser
        A Pandas Series comprising the adjusted close prices of a stock, identified
        by its ticker `tic`, for a specified date range. This range is inclusive,
        beginning from the `start` date and extending through to the `end` date.

        This data frame must meet the following criteria:
        - ser.index: `DatetimeIndex` with dates, matching the dates contained in
          the CSV file. The labels in the index must be `datetime` objects.

        - ser.columns: Rename the adjusted price column to `tic` in lowercase

        - ser.index.name: the index name remains the same as it is in the imported
          CSV file.

    Notes:
         - Ensure that the returned series does not contain any entries with null values.

    Examples
    --------
    IMPORTANT: The examples below are for illustration purposes. Your ticker/sample
    period may be different.

        >> tic = 'AAPL'
        >> ser = read_prc_csv(tic, '2010-01-04', '2010-12-31')
        >> util.test_print(ser)

    ----------------------------------------
    Date
    2010-01-04     6.604801
    2010-01-05     6.616219
                    ...
    2010-12-30     9.988830
    2010-12-31     9.954883
    Name: aapl, Length: 252, dtype: float64

    Obj type is: <class 'pandas.core.series.Series'>

    <class 'pandas.core.series.Series'>
    DatetimeIndex: 252 entries, 2010-01-04 to 2010-12-31
    Series name: aapl
    Non-Null Count  Dtype
    --------------  -----
    252 non-null    float64
    dtypes: float64(1)
    memory usage: 3.9 KB
    ----------------------------------------
     Hints
     -----
     - Remember that the ticker `tic` in `<tic>`_prc.csv is in lower case.
       File names in non-windows systems are case sensitive (so 'AAA.csv' and
       'aaa.csv' are different files).

    """
    # tic is lowercase
    tic = tic.lower();
    # pull data from csv using pandas and using cfg.DATADIR
    print(cfg.DATADIR)
    df = pd.read_csv(f'{cfg.DATADIR}/{tic}_prc.csv', index_col='Date', parse_dates=True)
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    filter = (df.index >= start) & (df.index <= end)
    df = df.loc[filter]
    # ensure no entries have null values
    if prc_col in df.columns:
        df = df[[prc_col]].dropna()
    else:
        raise ValueError(f"{prc_col} does not exist within df columns")

    # convert to df
    convertedSeries = df[prc_col]
    convertedSeries.name = tic
    # Sort dates in ascending order
    convertedSeries = convertedSeries.sort_index(ascending=True)
    return convertedSeries

# ----------------------------------------------------------------------------
# Part 4.3: Complete the daily_return_cal function
# ----------------------------------------------------------------------------
def daily_return_cal(prc: pd.Series) -> pd.Series:
    """ Create a pandas series containing daily returns for an individual stock,
    given a Pandas series with daily prices and datetime-formatted index.

    This function uses the `read_prc_csv` function in this module to read the
    price information for a stock.

    Parameters
    ----------
    prc : ser
        A Pandas series with adjusted closing price information for individual stocks.
        This series is the output of `read_prc_csv`.
        See the docstring of the `read_prc_csv` function for a description of this series.

    Returns
    -------
    ser,
        A series with daily return for a given ticker calculating from `prc` series.
        Daily returns are computed by dividing today's adjusted closing price to the
        previous trading day's adjusted closing price, minus 1.

        - ser.index: DatetimeIndex with dates. These dates should include all
          dates in the `prc` data frame excluding those with NaN daily return value.

        - ser.name: should be same as `prc` series

    Examples
    --------
    Note: The examples below are for illustration purposes. Your ticker/sample
    period may be different.

        >> ser_price = read_prc_csv(tic='AAPL', start='2020-09-03', end='2020-09-09')
        >> _test_daily_return_cal(made_up_data=False, ser_prc=ser_price)

        ----------------------------------------
        This is the test ser `prc`:

        Date
        2020-09-03    120.879997
        2020-09-04    120.959999
        2020-09-08    112.820000
        2020-09-09    117.320000
        Name: aapl, dtype: float64

        Obj type is: <class 'pandas.core.series.Series'>

        <class 'pandas.core.series.Series'>
        DatetimeIndex: 4 entries, 2020-09-03 to 2020-09-09
        Series name: aapl
        Non-Null Count  Dtype
        --------------  -----
        4 non-null      float64
        dtypes: float64(1)
        memory usage: 64.0 bytes
        ----------------------------------------

        ----------------------------------------
        This means `res_daily = daily_return_cal(prc)`, print out res_daily:

        Date
        2020-09-04    0.000662
        2020-09-08   -0.067295
        2020-09-09    0.039887
        Name: aapl, dtype: float64

        Obj type is: <class 'pandas.core.series.Series'>

        <class 'pandas.core.series.Series'>
        DatetimeIndex: 3 entries, 2020-09-04 to 2020-09-09
        Series name: aapl
        Non-Null Count  Dtype
        --------------  -----
        3 non-null      float64
        dtypes: float64(1)
         usage: 48.0 bytes
        ----------------------------------------

    Hints
     -----
     - Ensure that the returns do not contain any entries with null values.

    """
    daily_returns = prc.pct_change().dropna()
    daily_returns.name = prc.name  # Ensure the output series retains the same name as the input series

    return daily_returns


# ----------------------------------------------------------------------------
# Part 4.4: Complete the monthly_return_cal function
# ----------------------------------------------------------------------------
def monthly_return_cal(prc: pd.Series) -> pd.Series:
    """ Create a pandas series containing monthly returns for an individual stock,
    given a Pandas series with daily prices and datetime-formatted index.

    This function uses the `read_prc_csv` function in this module to read the
    price information .

    Parameters
    ----------
    prc : ser
        A Pandas series with adjusted closing price information for individual stocks.
        This series is the output of `read_prc_csv`.
        See the docstring of the `read_prc_csv` function for a description of this series.

    Returns
    -------
    ser,
        A series with monthly return for a given ticker calculating from `prc`.
        Monthly returns are computed by dividing the current end-of-month
        adjusted price to the previous end-of-month adjusted price, minus 1.
        If the number of price entries in `prc` of a year-month is less than 18,
        the calculated monthly return should **NOT** be presented in resulting ser

        - ser.index: PeriodIndex with year-month. These year-month should include all
        year-month from the prc series, but exclude months with fewer than 18 price entries.
        This exclusion criteria means that any month with less than 18 entries in the 'prc'
        series should not be considered.

        - ser.index.name: rename the series index as 'Year_Month'

        - ser.name: should be same as `prc` series

    Notes:
        - There are different ways to achieve the function goal.
          For this project, please read the documentation for the method pandas.Series.resample to
          convert a time series from daily to monthly.
          Read to_period() documentations to convert DatetimeIndex to PeriodIndex

    Examples
    --------
    Note: The examples below are for illustration purposes. Your ticker/sample
    period may be different.

        >> ser_price = read_prc_csv(tic='AAPL', start='2020-08-31', end='2021-01-10')
        >> _test_monthly_return_cal(made_up_data=False, ser_prc=ser_price)

        prc series:
        ----------------------------------------
        This is the test ser `prc`:

        Date
        2020-08-31    129.039993
        2020-09-01    134.179993
        ...
        2020-09-30    115.809998
        ...
        2020-10-15    120.709999
        2020-10-16    119.019997
        Name: aapl, dtype: float64

        Obj type is: <class 'pandas.core.series.Series'>

        <class 'pandas.core.series.Series'>
        DatetimeIndex: 34 entries, 2020-08-31 to 2020-10-16
        Series name: aapl
        Non-Null Count  Dtype
        --------------  -----
        34 non-null     float64
        dtypes: float64(1)
        memory usage: 544.0 bytes
        ----------------------------------------

        ----------------------------------------
        This means `res_monthly = monthly_return_cal(prc)`, print out res_monthly:

        Year_Month
        2020-09   -0.102526
        Freq: M, Name: aapl, dtype: float64

        Obj type is: <class 'pandas.core.series.Series'>

        <class 'pandas.core.series.Series'>
        PeriodIndex: 1 entries, 2020-09 to 2020-09
        Freq: M
        Series name: aapl
        Non-Null Count  Dtype
        --------------  -----
        1 non-null      float64
        dtypes: float64(1)
        memory usage: 16.0 bytes
        ----------------------------------------

    Hints
     -----
     - Ensure that the returns do not contain any entries with null values.

    """
    # resample the data to monthly, take the percent change and then drop all null vals.
    monthly_entry_counts = prc.resample('M').count()
    monthly_returns = prc.resample('M').last().pct_change().dropna()
    # Filter out months with fewer than 18 entries
    monthly_returns = monthly_returns[monthly_entry_counts >= 18]

    # Set the name and convert the index to PeriodIndex
    monthly_returns.name = prc.name
    monthly_returns.index = monthly_returns.index.to_period('M')
    monthly_returns.index.name = 'Year_Month'
    return monthly_returns


# ----------------------------------------------------------------------------
# Part 4.5: Complete the aj_ret_dict function
# ----------------------------------------------------------------------------

def aj_ret_monthly_df(stocks):
    monthly_df = pd.DataFrame()
    for src in stocks:
        if monthly_df.empty:
            monthly_df = src.to_frame()
        else:
            monthly_df = monthly_df.join(src.to_frame(), how='outer')
    return monthly_df

def aj_ret_daily_df(stocks):
    daily_df = pd.DataFrame()
    for src in stocks:
        if daily_df.empty:
            daily_df = src.to_frame()
        else:
            daily_df = daily_df.join(src.to_frame(), how='outer')
    return daily_df

def aj_ret_dict(tickers: list, start:str, end:str) -> dict:
    """ Create a dictionary with two items, each containing a dataframe
    of daily and monthly returns for all stocks listed in `tickers`.

    The keys for these items should be named 'Daily' and 'Monthly', respectively,
    resulting in the following structure:
        {"Daily": daily_return_df,
         "Monthly": monthly_return_df}

    This function uses the `read_prc_csv` function from this module to retrieve
    price information over a given time range, and then applies the daily_return_cal
    and monthly_return_cal functions to compute daily and monthly returns for each ticker
    listed in `tickers`. Finally, it converts the series of daily and monthly returns
    into dataframes and stores them in a dictionary, with 'Daily' and 'Monthly'
    serving as the keys for their respective dataframes.

    Parameters
    ----------
    tickers : list
        A list containing all the stock tickers that users intend to process through
        this function.
        String of the tickers inside the 'tickers' can include lowercase and/or
        uppercase characters
    start  :  str
        The inclusive start date for the date range when retrieve price information
        from CSV file. It is a parameter will be used in `read_prc_csv` function.
        For example: if you enter '2010-09-02', the `read_prc_csv` function will include
        data from this date onwards.
        And please make sure the provided start date is a valid calendar date.
    end  :  str
        The inclusive end date for the date range when retrieve price information
        from CSV file. It is a parameter will be used in `read_prc_csv` function.
        For example: if you enter '2010-12-20', the `read_prc_csv` function output series
        will encompass data up to and including December 20, 2010.
        And please make sure the provided start date is a valid calendar date.

    Returns
    -------
    dic,
        A dictionary with two items, each containing a dataframe of daily and monthly returns
        for all stocks listed in the 'tickers' list.

        - dic.keys(): The dictionary should have two keys: "Daily" and "Monthly".

        - dic.values(): The dictionary should have two {key:value} pairs.
          The "Daily"/"Monthly" key contains a dataframe of daily/monthly returns for all stocks
          listed in the 'tickers' list

        - dic['Daily'].index/dic['Monthly'].index: DatetimeIndex/PeriodIndex with dates/year-month.
          The dates/year-month should include all those in the return series outputted by
          the daily_return_cal/monthly_return_cal function, each generated for the respective
          tickers in the `tickers` list.

        - dic['Daily'].columns/dic['Monthly'].columns: should include all tickers from the
          tickers list, converted to lowercase.

        - dic['Daily'].index.name/dic['Monthly'].index.name: 'Date'/'Year_Month'

    Examples:
    --------
    Note: The examples below are for illustration purposes. Your ticker/sample
    period may be different.
        If we run _test_aj_ret_dict(['AAPL', 'TSLA'], start='2010-06-25', end='2010-08-05'),
        it will print out:

        ----------------------------------------
        This means `dict_ret = aj_ret_dict(tickers, start, end)`, print out dict_ret:

        {'Daily':                 aapl      tsla
        Date
        2010-06-28  0.006000       NaN
        2010-06-29 -0.045211       NaN
        2010-06-30 -0.018113 -0.002512
        2010-07-01 -0.012126 -0.078472
        ...
        2010-08-04  0.004009 -0.031435
        2010-08-05 -0.004867 -0.038100,
         'Monthly':                 aapl     tsla
        Year_Month
        2010-07     0.022741 -0.16324}

        Obj type is: <class 'dict'>

        the Key of the dictionary is Daily, value info:
        <class 'pandas.core.frame.DataFrame'>
        DatetimeIndex: 28 entries, 2010-06-28 to 2010-08-05
        Data columns (total 2 columns):
         #   Column  Non-Null Count  Dtype
        ---  ------  --------------  -----
         0   aapl    28 non-null     float64
         1   tsla    26 non-null     float64
        dtypes: float64(2)
        memory usage: 672.0 bytes

        the Key of the dictionary is Monthly, value info:
        <class 'pandas.core.frame.DataFrame'>
        PeriodIndex: 1 entries, 2010-07 to 2010-07
        Freq: M
        Data columns (total 2 columns):
         #   Column  Non-Null Count  Dtype
        ---  ------  --------------  -----
         0   aapl    1 non-null      float64
         1   tsla    1 non-null      float64
        dtypes: float64(2)
        memory usage: 24.0 bytes
        ----------------------------------------
    """
    # OWN NOTES
    '''
    Create a dict with two keys, containing daily and monthly returns for given
    list of tickers (can assume list will not be empty?)

    STRUCTURE:
    dict: {
        "Daily": daily_ret_df,
        "Monthly": monthly_return_df
    }

    - USE: read_prc_csv
    - USE: daily_ret_cal + monthly_ret_cal
    '''

# Create list of dataframes to manipulate from ticker
# Init dict with dummy values
    daily_returns = []
    monthly_returns = []
    for ticker in tickers:
        price_data = read_prc_csv(ticker.lower(), start, end)
        daily_returns.append(daily_return_cal(price_data).rename(ticker.lower()))
        monthly_returns.append(monthly_return_cal(price_data).rename(ticker.lower()))

    return {
        'Daily':  aj_ret_daily_df(stocks=daily_returns),
        'Monthly': aj_ret_monthly_df(stocks=monthly_returns)
    }


# ----------------------------------------------------------------------------
#   Test functions
# ----------------------------------------------------------------------------


def _test_read_prc_csv():
    """ Test function for `read_prc_csv`
    """
    tic = 'AAPL'
    ser = read_prc_csv(tic, '2010-01-04', '2010-12-31')
    util.test_print(ser)

    # AAPL data for 2010-12-31 to 2010-01-04

def _test_daily_return_cal(made_up_data=True, ser_prc=None):
    """ Test function for `daily_return_cal`
    """
    if made_up_data:
        # Made-up data
        prc = pd.Series({
             '2019-01-29': 1.0, '2019-01-30': 2.0, '2019-01-31': 1.5, '2019-02-01': 1.4,
             '2019-02-04': 1.6, '2019-02-05': 1.2, '2019-02-06': 1.4, '2019-02-07': 1.9, '2019-02-08': 1.7, })
        prc.name = 'comp_tic'
        prc.index = pd.to_datetime(prc.index)
        prc.index.name = 'Date'
    else:
        prc = ser_prc.copy()

    msg = 'This is the test ser `prc`:'
    util.test_print(prc, msg)

    res_daily = daily_return_cal(prc)
    msg = "This means `res_daily = daily_return_cal(prc)`, print out res_daily:"
    util.test_print(res_daily, msg)


def test_daily_ret_test():
    prc1 = daily_return_cal(read_prc_csv('TSLA', start='2010-06-25', end='2010-08-05'))
    print(prc1)


def _test_monthly_return_cal(made_up_data=True, ser_prc=None):
    """ Test function for `monthly_return_cal`
    """
    if made_up_data:
        # Made-up data
        prc = pd.Series({
              '2019-01-28': 2.0, '2019-01-29': 2.0, '2019-01-30': 1.0, '2019-01-31': 1.5, '2019-02-01': 1.4,
              '2019-02-04': 1.6, '2019-02-05': 1.2, '2019-02-06': 1.4, '2019-02-07': 1.9, '2019-02-08': 1.7,
              '2019-02-11': 1.7, '2019-02-12': 1.5, '2019-02-13': 1.7, '2019-02-14': 1.2, '2019-02-15': 1.4,
              '2019-02-18': 0.9, '2019-02-19': 1.4, '2019-02-20': 1.7, '2019-02-21': 1.0, '2019-02-22': 1.2,
              '2019-02-25': 0.8, '2019-02-26': 1.7, '2019-02-27': 1.4, '2019-02-28': 1.0, '2019-03-01': 1.3, })
        prc.name = 'comp_tic'
        prc.index = pd.to_datetime(prc.index)
        prc.index.name = 'Date'
    else:
        prc = ser_prc.copy()

    msg = 'This is the test ser `prc`:'
    util.test_print(prc, msg)

    res_monthly = monthly_return_cal(prc)
    msg = "This means `res_monthly = monthly_return_cal(prc)`, print out res_monthly:"
    util.test_print(res_monthly, msg)


def _test_aj_ret_dict(tickers, start, end):
    """ Test function for `aj_ret_dict`
    """

    dict_ret = aj_ret_dict(tickers, start, end)

    msg = "This means `dict_ret = aj_ret_dict(tickers, start, end)`, print out dict_ret:"
    util.test_print(dict_ret, msg)

    return dict_ret

'''
 UNIT TEST SUITE
'''
class functionalityTests(unittest.TestCase):
    def test_read_prc_csv(self):
        tic = 'AAPL'
        ser = read_prc_csv(tic, '2010-01-04', '2010-12-31')
        self.assertEqual(ser.index[0], pd.to_datetime('2010-01-04'))
        self.assertEqual(ser.index[-1], pd.to_datetime('2010-12-31'))
        # Value check at the beginning and end
        '''
        2010-01-04     6.604801
        2010-01-05     6.616219
                    ...
        2010-12-30     9.988830
        2010-12-31     9.954883
        '''
        # Assert first 2 and last 2 are equal to the expected values
        self.assertEqual(round(ser.iloc[0], 6), 6.604801)
        self.assertEqual(round(ser[pd.Timestamp('2010-01-05')], 6), 6.616219)
        self.assertEqual(round(ser[pd.Timestamp('2010-12-30')], 6), 9.988830)
        self.assertEqual(round(ser.iloc[-1], 6), 9.954883)
        self.assertEqual(ser.name, 'aapl')

    def test_daily_return_cal(self):
        tic = 'AAPL'
        ser_price = read_prc_csv(tic='AAPL', start='2020-09-03', end='2020-09-09')
        res_daily = daily_return_cal(ser_price)
        '''
        Testing scenario found in docsstring
        Date
        2020-09-04    0.000662
        2020-09-08   -0.067295
        2020-09-09    0.039887
        Name: aapl, dtype: float64
        '''
        # calculate for a 1 day return
        dates = pd.to_datetime(['2020-09-04', '2020-09-08', '2020-09-09'])
        values = [0.000662, -0.067295, 0.039887]

# Create the Series with the specified dates as the index
        expected = pd.Series(values, index=dates)
        expected.index.name = 'Date'
        expected.name = 'aapl'
        res_daily = res_daily.round(6)
        pd.testing.assert_series_equal(res_daily, expected)

    def test_monthly_return_cal_made_up_data(self):
        '''
        Converted given test with made up data to a unit test
        '''
        prc = pd.Series({
              '2019-01-28': 2.0, '2019-01-29': 2.0, '2019-01-30': 1.0, '2019-01-31': 1.5, '2019-02-01': 1.4,
              '2019-02-04': 1.6, '2019-02-05': 1.2, '2019-02-06': 1.4, '2019-02-07': 1.9, '2019-02-08': 1.7,
              '2019-02-11': 1.7, '2019-02-12': 1.5, '2019-02-13': 1.7, '2019-02-14': 1.2, '2019-02-15': 1.4,
              '2019-02-18': 0.9, '2019-02-19': 1.4, '2019-02-20': 1.7, '2019-02-21': 1.0, '2019-02-22': 1.2,
              '2019-02-25': 0.8, '2019-02-26': 1.7, '2019-02-27': 1.4, '2019-02-28': 1.0, '2019-03-01': 1.3, })
        prc.name = 'comp_tic'
        prc.index = pd.to_datetime(prc.index)
        prc.index.name = 'Date'
        # Make monthly return series
        res_monthly = monthly_return_cal(prc)
        expected_returns = pd.Series([-0.333333], index=pd.PeriodIndex(['2019-02'], name='Year_Month', freq='M'))
        expected_returns.name = 'comp_tic'
        pd.testing.assert_series_equal(res_monthly, expected_returns)


    def test_monthly_return_data(self):
        '''
        Testing scenario found in function docstring:
        Date
        2020-08-31    129.039993
        2020-09-01    134.179993
        ...
        2020-09-30    115.809998
        ...
        2020-10-15    120.709999
        2020-10-16    119.019997
        Name: aapl, dtype: float64
        '''
        ser_price = read_prc_csv(tic='AAPL', start='2020-08-31', end='2021-01-10')
        output = monthly_return_cal(ser_price)
        # Assert that our output is the same for all the given dates
        self.assertEqual(round(ser_price[pd.Timestamp('2020-08-31')], 6), 129.039993)
        self.assertEqual(round(ser_price[pd.Timestamp('2020-09-01')], 6), 134.179993)
        self.assertEqual(round(ser_price[pd.Timestamp('2020-09-30')], 6), 115.809998)
        self.assertEqual(round(ser_price[pd.Timestamp('2020-10-15')], 6), 120.709999)
        self.assertEqual(round(ser_price[pd.Timestamp('2020-10-16')], 6), 119.019997)

    def test_aj_ret_dict(self):
        '''
        Testing Scenario:
        input : (['AAPL', 'TSLA'], start='2010-06-25', end='2010-08-05')

        {'Daily':                 aapl      tsla
        Date
                    2010-06-28  0.006000       NaN
                    2010-06-29 -0.045211       NaN
                    2010-06-30 -0.018113 -0.002512
                    2010-07-01 -0.012126 -0.078472
        ...
                    2010-08-04  0.004009 -0.031435
                    2010-08-05 -0.004867 -0.038100,

         'Monthly':                 aapl     tsla
        Year_Month
                    2010-07     0.022741 -0.16324}

        (Can only test one year_month because only one is given)
        '''
        output = aj_ret_dict(['AAPL', 'TSLA'], start='2010-06-25', end='2010-08-05')
        self.assertEqual(round(output['Daily'].loc[pd.Timestamp('2010-06-28')]['aapl'], 6), 0.006000)
        self.assertTrue(np.isnan(output['Daily'].loc[pd.Timestamp('2010-06-28')]['tsla']))
        self.assertEqual(round(output['Daily'].loc[pd.Timestamp('2010-06-29')]['aapl'], 6), -0.045211)
        self.assertTrue(np.isnan(output['Daily'].loc[pd.Timestamp('2010-06-29')]['tsla']))
        self.assertEqual(round(output['Daily'].loc[pd.Timestamp('2010-06-30')]['aapl'], 6), -0.018113)
        self.assertEqual(round(output['Daily'].loc[pd.Timestamp('2010-06-30')]['tsla'], 6), -0.002512)
        self.assertEqual(round(output['Daily'].loc[pd.Timestamp('2010-07-01')]['aapl'], 6), -0.012126)
        self.assertEqual(round(output['Daily'].loc[pd.Timestamp('2010-07-01')]['tsla'], 6), -0.078472)
        self.assertEqual(round(output['Daily'].loc[pd.Timestamp('2010-08-04')]['aapl'], 6), 0.004009)
        self.assertEqual(round(output['Daily'].loc[pd.Timestamp('2010-08-04')]['tsla'], 6), -0.031435)
        self.assertEqual(round(output['Daily'].loc[pd.Timestamp('2010-08-05')]['aapl'], 6), -0.004867)
        self.assertEqual(round(output['Daily'].loc[pd.Timestamp('2010-08-05')]['tsla'], 6), -0.038100)
        self.assertEqual(round(output['Monthly'].loc[pd.Period('2010-07', freq='M')]['aapl'], 6), 0.022741)
        self.assertEqual(round(output['Monthly'].loc[pd.Period('2010-07', freq='M')]['tsla'], 6), -0.16324)


if __name__ == "__main__":
    # #test read_prc_csv function
    # _test_read_prc_csv()

    # # use made-up series to test daily_return_cal function
    # _test_daily_return_cal()
    # # use AAPL prc series to test daily_return_cal function
    # ser_price = read_prc_csv(tic='AAPL', start='2020-09-03', end='2020-09-09')
    # _test_daily_return_cal(made_up_data=False, ser_prc=ser_price)
    #
    # # use made-up series to test daily_return_cal function
    # _test_monthly_return_cal()
    # # use AAPL prc series to test daily_return_cal function
    # ser_price = read_prc_csv(tic='AAPL', start='2020-08-31', end='2021-01-10')
    # _test_monthly_return_cal(made_up_data=False, ser_prc=ser_price)
    # # test aj_ret_dict function
    #_test_aj_ret_dict(['AAPL', 'TSLA'], start='2010-06-25', end='2010-08-05')
    #test_daily_ret_test()
    unittest.main()
    pass


