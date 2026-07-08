import unittest

from alphapy import data


class TestFXMacroDataData(unittest.TestCase):

    def test_get_fxmacrodata_data(self):
        class MockResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    'data': [
                        {'date': '2026-01-02', 'val': 1.2},
                        {'date': '2026-01-01', 'val': 1.1},
                    ]
                }

        calls = {}

        def mock_get(url, params):
            calls['url'] = url
            calls['params'] = params
            return MockResponse()

        original_get = data.requests.get
        try:
            data.requests.get = mock_get
            df = data.get_fxmacrodata_data(
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
            data.requests.get = original_get

        self.assertEqual(calls['url'], 'https://fxmacrodata.com/api/v1/forex/EUR/USD')
        self.assertEqual(calls['params']['start_date'], '2026-01-01')
        self.assertEqual(list(df.columns), ['date', 'open', 'high', 'low', 'close', 'volume'])
        self.assertEqual(list(df['close']), [1.2, 1.1])


if __name__ == '__main__':
    unittest.main()
