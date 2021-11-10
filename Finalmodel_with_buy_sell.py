def Stock_Prediction(URL):
    import pandas as pd
    import numpy as np
    import re
    import json
    import csv
    from io import StringIO
    from bs4 import BeautifulSoup
    import requests
    from fbprophet import Prophet
    params = {
        'range': '5y',
        'interval': '1d',
        'events': 'history'
    }
    response = requests.get(URL, params=params)
    file = StringIO(response.text)
    reader = csv.reader(file)
    data = pd.DataFrame(reader)
    train = pd.DataFrame({'ds': data[0].iloc[1:], 'y': data[4].iloc[1:]})
    train['ds'] = pd.to_datetime(train['ds'])
    New_month = []
    for i in pd.DatetimeIndex(train['ds']).day:
        if (i > 5):
            New_month.append(0)
        else:
            New_month.append(1)
    New_year = []
    for i, j in zip(pd.DatetimeIndex(train['ds']).day, pd.DatetimeIndex(train['ds']).month):
        if (i == 31 and j == 12):
            New_year.append(1)

        elif (i >= 1 and i < 5 and j == 1):
            New_year.append(1)

        else:
            New_year.append(0)
    train['New_year'] = New_year
    train['New_month'] = New_month
    m = Prophet(changepoint_prior_scale=3, holidays_prior_scale=0.001, changepoint_range=1, n_changepoints=150)
    m.add_country_holidays(country_name='IN')
    m.add_regressor('New_year')
    m.add_regressor('New_month')
    m.fit(train)
    future = m.make_future_dataframe(periods=30)
    for i in pd.DatetimeIndex(future['ds'].iloc[len(train):]).day:
        if (i > 5):
            New_month.append(0)
        else:
            New_month.append(1)
    for i, j in zip(pd.DatetimeIndex(future['ds'].iloc[len(train):]).day,
                    pd.DatetimeIndex(future['ds'].iloc[len(train):]).month):
        if (i == 31 and j == 12):
            New_year.append(1)

        elif (i >= 1 and i < 5 and j == 1):
            New_year.append(1)

        else:
            New_year.append(0)

    df = pd.DataFrame({'ds': future['ds'], 'New_year': New_year, 'New_month': New_month})
    forecast = m.predict(df)
    lower = []
    upper = []
    for i, j in zip(forecast['yhat_lower'], forecast['yhat']):
        lower.append(j - i)
    for i, j in zip(forecast['yhat_upper'], forecast['yhat']):
        upper.append(i - j)

    avg = []
    for i, j in zip(lower, upper):
        avg.append((i + j) / 2)

    error = []
    for i, j in zip(forecast['yhat'].iloc[len(train):], avg[len(train):]):
        print(i, '+/-', j)
        error.append(j / i)
    forecast['MA'] = forecast['yhat'].rolling(30).mean().shift()
    what_to_do = []
    for i, j in zip(forecast['yhat'], forecast['MA']):
        if (i == np.NaN or j == np.NaN):
            what_to_do.append(np.nan)
        elif (i > j):
            what_to_do.append('BUY')
        else:
            what_to_do.append('SELL')

    return m.plot(forecast), pd.DataFrame(
        {'Date': future['ds'].iloc[len(train):], 'Close': forecast['yhat'].iloc[len(train):],
         'Tolerance': avg[len(train):], 'MA': forecast['MA'].iloc[len(train):], 'BUY/SELL': what_to_do[len(train):]}), np.array(error).mean(), max(error), min(error)