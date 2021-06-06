#!/usr/bin/python3
"""
Basics of pandas data structures.

DataFrames are like spreadsheets. Data is stored in cells. Data has named
columns and numbered rows.
"""

import numpy as np
import pandas as pd


arr = np.random.randint(low=0, high=50, size=(5,2))
cols = ['temperature', 'activity']
# Compose DataFrame from simple arrays.
df = pd.DataFrame(data=arr, columns=cols)

# Can edit columns via broadcasting.
df['adjusted'] = df['activity']*2

# DataFrames are composed of Series.
print(df)

# DataFrame subsets. Get rows, columns, slices, or cells of a DataFrame.
# Rows 0-1
print(df.head(2))

# Print row #2:
# Integer based location by position.
print(df.iloc[[2]]) # As a DataFrame
print(df.iloc[2]) # As a Series

# Print slice
print(df[1:4])

# Print column
print(df['temperature'])

# Duplicating DataFrames
# Copy by reference by assigning to other variable
dfc = df

# Copy by value.
dfc = df.copy()