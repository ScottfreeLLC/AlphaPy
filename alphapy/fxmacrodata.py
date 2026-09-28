################################################################################
#
# Package   : AlphaPy
# Module    : fxmacrodata
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
################################################################################

"""FXMacroData daily FX reference-rate adapter for MarketFlow."""

import logging
import os

import pandas as pd
import requests


logger = logging.getLogger(__name__)

FXMACRODATA_API_ROOT = 'https://api.fxmacrodata.com/v1'
FXMACRODATA_PAGE_SIZE = 100


def get_fxmacrodata_data(schema, subschema, symbol, intraday_data, data_fractal,
                         from_date, to_date, lookback_period):
    r"""Get daily FX reference rates from FXMacroData.

    The parameters match AlphaPy's market-data dispatch contract. Native
    reference-observation OHLC is preserved when present; otherwise the daily
    reference value is copied into OHLC and volume is set to zero.

    """

    df = pd.DataFrame()
    if intraday_data:
        logger.info("FXMacroData supports daily reference rates, not intraday bars")
        return df

    pair = symbol.upper().replace('/', '').replace('-', '').replace('_', '')
    if len(pair) != 6 or not pair.isalpha() or not pair.isascii():
        logger.error("FXMacroData symbol must be formatted like EURUSD or EUR/USD")
        return df

    base = pair[:3]
    quote = pair[3:]
    url = '/'.join([FXMACRODATA_API_ROOT.rstrip('/'), 'forex', base, quote])
    base_params = {'start_date': from_date, 'end_date': to_date}
    headers = {}
    api_key = (os.environ.get('FXMACRODATA_API_KEY') or
               os.environ.get('FXMD_API_KEY'))
    if api_key:
        headers['X-API-Key'] = api_key

    rows = []
    offset = 0
    try:
        while True:
            params = dict(base_params)
            params.update({'limit': FXMACRODATA_PAGE_SIZE, 'offset': offset})
            response = requests.get(url, params=params, headers=headers, timeout=30)
            if not response.ok:
                logger.info("FXMacroData returned HTTP %s for %s",
                            response.status_code, symbol.upper())
                return df
            payload = response.json()
            page = payload.get('data', []) if isinstance(payload, dict) else []
            if not isinstance(page, list):
                logger.info("FXMacroData returned invalid data for %s", symbol.upper())
                return df
            rows.extend(row for row in page if isinstance(row, dict))
            if len(page) < FXMACRODATA_PAGE_SIZE:
                break
            offset += FXMACRODATA_PAGE_SIZE
    except (requests.RequestException, ValueError, TypeError):
        logger.info("Could not retrieve %s data with FXMacroData", symbol.upper())
        return df

    records = []
    for row in rows:
        try:
            value = float(row['val'])
            record = (
                row['date'],
                float(row.get('open', value)),
                float(row.get('high', value)),
                float(row.get('low', value)),
                float(row.get('close', value)),
                0.0,
            )
        except (KeyError, TypeError, ValueError):
            continue
        records.append(record)

    if records:
        df = pd.DataFrame.from_records(
            records,
            columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        df.drop_duplicates(subset=['date'], keep='first', inplace=True)
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)

    return df
