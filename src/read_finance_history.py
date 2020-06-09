
import os
import pandas as pd

DATASET_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")

# https://finance.yahoo.com/quote/SPY/history?p=SPY
DEFAULT_FILES = [os.path.join(DATASET_ROOT, "SPY.csv"),
                 os.path.join(DATASET_ROOT, "TLT.csv"),
                 os.path.join(DATASET_ROOT, "TIGER200.csv"),
                 os.path.join(DATASET_ROOT, "USD-KRW.csv")]


def read_files(csv_files=DEFAULT_FILES, start="2010-05-01", end="2020-05-29"):
    dates = pd.date_range(start, end)
    titles = [f[f.rfind("/")+1:f.find(".csv")] for f in csv_files]

    # create empty df
    df = pd.DataFrame(index=dates)
    for title, filename in zip(titles, csv_files):
        df_tmp = pd.read_csv(filename, usecols=["Date", "Adj Close"], na_values=["nan"])
        df_tmp = df_tmp.set_index("Date")
        df_tmp = df_tmp.rename(columns={"Adj Close": title})
        df = df.join(df_tmp)
        df = df.dropna()
    return df


if __name__ == "__main__":
    df = read_files()
    print(df.head())

