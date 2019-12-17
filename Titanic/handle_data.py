import numpy as np

def convert_non_numerical(df):
    cols = df.columns.values
    for col in cols:
        values = {}
        def convert2int(val):
            return values[val]
        if df[col].dtype != np.int64 and df[col].dtype != np.float64: # if not numerical
            col_values = set(df[col].values.tolist())
            i = 0
            for value in col_values:
                if value not in values: # new value
                    values[value] = i
                    i += 1
            df[col] = list(map(convert2int, df[col]))
    return df