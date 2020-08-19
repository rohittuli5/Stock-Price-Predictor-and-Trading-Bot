from alpha_vantage.timeseries import TimeSeries
from pprint import pprint

def save_dataset(symbol, time_window):
    api_key=''
    print(symbol, time_window)
    ts = TimeSeries(key=api_key, output_format='pandas')
    data, meta_data = ts.get_daily(symbol, outputsize='full')


    pprint(data.head(10))

    data.to_csv(f'./{symbol}_{time_window}.csv')


if __name__ == "__main__":
    save_dataset('MSFT','daily')
