import unittest

from alphapy import fxmacrodata


class TestFXMacroDataData(unittest.TestCase):

    def test_get_fxmacrodata_data(self):
        class MockResponse:
            ok = True
            status_code = 200

            def json(self):
                return {
                    'data': [
                        {'date': '2026-01-02', 'val': 1.2},
                        {'date': '2026-01-01', 'val': 1.1},
                    ]
                }

        calls = {}

        def mock_get(url, params, timeout):
            calls['url'] = url
            calls['params'] = params
            calls['timeout'] = timeout
            return MockResponse()

        original_get = fxmacrodata.requests.get
        try:
            fxmacrodata.requests.get = mock_get
            df = fxmacrodata.get_fxmacrodata_data(
                'fxmacrodata',
                '',
                'EUR/USD',
                False,
                '1D',
                '2026-01-01',
                '2026-01-02',
                2,
            )
        finally:
            fxmacrodata.requests.get = original_get

        self.assertEqual(calls['url'], 'https://api.fxmacrodata.com/v1/forex/EUR/USD')
        self.assertEqual(calls['params']['start_date'], '2026-01-01')
        self.assertEqual(calls['params']['limit'], 100)
        self.assertEqual(calls['timeout'], 30)
        self.assertEqual(list(df.columns), ['date', 'open', 'high', 'low', 'close', 'volume'])
        self.assertEqual(list(df['close']), [1.1, 1.2])

    def test_get_fxmacrodata_data_preserves_ohlc_and_paginates(self):
        calls = []

        class MockResponse:
            ok = True
            status_code = 200

            def __init__(self, rows):
                self.rows = rows

            def json(self):
                return {'data': self.rows}

        first_page = [
            {'date': '2026-01-02', 'val': 1.2,
             'open': 1.1, 'high': 1.3, 'low': 1.0, 'close': 1.25}
        ] * 100
        second_page = [{'date': '2026-01-01', 'val': '1.05'}]

        def mock_get(url, params, timeout):
            calls.append(dict(params))
            rows = first_page if params['offset'] == 0 else second_page
            return MockResponse(rows)

        original_get = fxmacrodata.requests.get
        try:
            fxmacrodata.requests.get = mock_get
            df = fxmacrodata.get_fxmacrodata_data(
                'fxmacrodata', '', 'EURUSD', False, '1D',
                '2026-01-01', '2026-01-02', 2)
        finally:
            fxmacrodata.requests.get = original_get

        self.assertEqual([call['offset'] for call in calls], [0, 100])
        self.assertEqual(list(df['date']), ['2026-01-01', '2026-01-02'])
        self.assertEqual(df.iloc[1]['open'], 1.1)
        self.assertEqual(df.iloc[1]['high'], 1.3)
        self.assertEqual(df.iloc[1]['low'], 1.0)
        self.assertEqual(df.iloc[1]['close'], 1.25)

    def test_get_fxmacrodata_data_rejects_invalid_pair_without_request(self):
        original_get = fxmacrodata.requests.get
        try:
            fxmacrodata.requests.get = lambda *args, **kwargs: self.fail('unexpected request')
            df = fxmacrodata.get_fxmacrodata_data(
                'fxmacrodata', '', 'EUR1USD', False, '1D',
                '2026-01-01', '2026-01-02', 2)
        finally:
            fxmacrodata.requests.get = original_get

        self.assertTrue(df.empty)


if __name__ == '__main__':
    unittest.main()
