""" to import password protected excel file
needs some more work to function properly """

import sys
import win32com.client

xlApp = win32com.client.Dispatch("Excel.Application")
print("Excel library version:", xlApp.Version)
filename = 'D:/Python/notebooks/ABI/DEV/MAIN/month_datasets/NEW/oct29_salary_insights_ask/US-Banded-Empl-Comp-Q3 2018- Final.xlsx' 
password = 'hrops'
xlwb = xlApp.Workbooks.Open(filename, Password=password)

xlws = xlwb.Sheets(2)

# Get last_row
row_num = 0
cell_val = ''
while cell_val != None:
    row_num += 1
    cell_val = xlws.Cells(row_num, 1).Value
    # print(row_num, '|', cell_val, type(cell_val))
last_row = row_num - 1
print(last_row)

# Get last_column
col_num = 0
cell_val = ''
while cell_val != None:
    col_num += 1
    cell_val = xlws.Cells(1, col_num).Value
    # print(col_num, '|', cell_val, type(cell_val))
last_col = col_num - 1
print(last_col)

# Get the content in the rectangular selection region
# content is a tuple of tuples
content = xlws.Range(xlws.Cells(1, 1), xlws.Cells(last_row, last_col)).Value 

# Transfer content to pandas dataframe
dataframe = pandas.DataFrame(list(content))