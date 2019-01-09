#Part 1: Dataset Exploration

#Consider the hurricane dataset at:
#https://web.archive.org/web/20080921102626/http://www.aoml.noaa.gov/hrd/hurdat/ushurrlist18512007.txt
#Explore the file using python. Answer the following:
#Show appropriate summary statistics and visualizations for: Months, Highest_Category, Central_Pressure_mb, Max_Winds_kt
#Show the relationship between Highest_Category and Max_Winds_kt Central_Pressure_mb and Max_Winds_kt
#Explain how you accounted for any missing data, and how that may have affected your results


import csv, os, math, operator, pickle, time, random, re

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
#%matplotlib inline

from collections import Counter
import statistics as stat

data_dir = "C:\PythonData"

os.chdir(data_dir)


##Replace spaces(more than 3) with a tab
def ReplaceSpacesWithTab(infile, outfile):
   infopen = open(infile, 'r', encoding="utf-8")
   outfopen = open(outfile, 'w', encoding="utf-8")

   lines = infopen.readlines()
   for line in lines:
      line = line.strip()
      line_change = re.sub("\s\s\s\s+", "\t", line)
      outfopen.writelines(line_change)
      outfopen.write('\n')
   infopen.close()
   outfopen.close()

ReplaceSpacesWithTab("./ushurrlist18512007.txt", "ushurrlist18512007_adjust.txt")

##Read ushurrlist18512007_adjust.txt
ColumnNames = ['Year', 'Month', 'Highest_Category', 'Central_Pressure', 'Max_Winds']
df1 = pd.read_table('./ushurrlist18512007_adjust.txt', skiprows = 4, usecols = [0, 1, 3, 4, 5],
                   sep = r'\t{1,}', engine = 'python', header = None, names = ColumnNames, parse_dates=['Year'])

##Covert columns ['Year','Highest_Category'] to numeric
df1.Year = pd.to_numeric(df1.Year, errors='coerce')
df1.Highest_Category = pd.to_numeric(df1.Highest_Category, errors = 'coerce')

##Delete rows which value is greater than 5 in column 'Highest_Category'
df2 = df1[df1['Highest_Category'] <= 5]

##Convert abnormal strings to NaN
df2.replace(['No-De','-----', '----- ', '-----   Helene'], np.nan, inplace = True)

##Remove strings(&,#) from column 'Month'
df2.replace({'Month':'&|#'}, '', regex = True, inplace = True)
df2.to_csv('df2.csv')
##print(df2)

##Read df2.csv
df3 = pd.read_table('./df2.csv', sep = ',', usecols = [1,2,3,4,5], skipinitialspace = True, header = 0, parse_dates = ['Year'])
df3.set_index('Year', inplace = True)

##Delete rows with 2 or more NA in columns['Month', 'Highest_Category', 'Central_Pressure', 'Max_Winds'].
## (Delete the row of 1925 which only has data in 'Highest_Category')
df4 = df3.dropna(thresh = 2)

#Convert strings to the corresponding numbers in 'Month'
df5 = df4.replace({'Month':' '}, '', regex = True)
df5 = df5.replace(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],
                 [1,2,3,4,5,6,7,8,9,10,11,12])
#df5.to_csv('df5.csv')

##Question - b
##Show appropriate summary statistics and visualizations for:
## Months, Highest_Category, Central_Pressure_mb, Max_Winds_kt
df5_withoutNA = df5.dropna()
df5_withoutNA.to_csv('ushurrlist18512007_WithoutNA.csv')

print(df5_withoutNA.describe())    #calculate mean values of Central_Pressue and Max_Winds
print(df5_withoutNA.dtypes)

##Calculate Frequency of 'Month'
Count_Month = Counter(df5_withoutNA.Month)
print(Count_Month)

##Plot histogram for 'Month'
possibleMonth = [6, 7, 8, 9, 10, 11]
fin = [possibleMonth.index(i) for i in df5_withoutNA.Month]
plt.hist(fin, bins = range(7), align = "left", edgecolor = 'black', color = 'cyan', histtype = 'bar')
plt.xticks(range(6), possibleMonth)
plt.grid(axis = 'y', alpha = 0.25)
plt.title("Month")
plt.xlabel("Month")
plt.ylabel("Frequency")
plt.show()

##Calculate Frequency of 'Highest_Category'
Count_Category = Counter(df5_withoutNA.Highest_Category)
print(Count_Category)

##Plot histogram for 'Highest_Category'
possibleCategory = [1, 2, 3, 4, 5]
fin = [possibleCategory.index(i) for i in df5_withoutNA.Highest_Category]
plt.hist(fin, bins = range(6), align = "left", edgecolor = 'black', color = 'cyan', histtype = 'bar')
plt.xticks(range(5), possibleCategory)
plt.grid(axis = 'y', alpha = 0.25)
plt.title("Highest Saffir-Simpson Category")
plt.xlabel("Highest Saffir-Simpson Category")
plt.ylabel("Frequency")
plt.show()

##Plot histogram for 'Central_Pressure'
plt.hist(df5_withoutNA.Central_Pressure, edgecolor='black', color='cyan', histtype='bar')
plt.grid(axis = 'y', alpha = 0.25)
plt.title("Histogram: Central Pressue")
plt.xlabel("Central Pressure")
plt.ylabel("Frequency")
plt.show()

##plot boxplot for 'Central_Pressure'
plt.boxplot(df5_withoutNA.Central_Pressure)
plt.grid(axis = 'y', alpha = 0.25)
plt.title("Boxplot: Central Pressue")
plt.xlabel("Central Pressure")
plt.ylabel("Values")
plt.show()

##Plot histogram for 'Max_Winds'
plt.hist(df5_withoutNA.Max_Winds, edgecolor='black', color='cyan', histtype='bar')
plt.grid(axis = 'y', alpha = 0.25)
plt.title("Max.Winds")
plt.xlabel("Max.Winds")
plt.ylabel("Frequency")
plt.show()

##plot boxplot for 'Max_Winds'
plt.boxplot(df5_withoutNA.Max_Winds)
plt.grid(axis = 'y', alpha = 0.25)
plt.title("Boxplot: Max.Winds")
plt.xlabel("Max.Winds")
plt.ylabel("Values")
plt.show()

##Question - c
##Show the relationship between Highest_Category and Max_Winds_kt Central_Pressure_mb and Max_Winds_kt

##Relationship between 'Highest_Category' and 'Central Pressure'
plt.scatter(df5_withoutNA.Highest_Category,df5_withoutNA.Central_Pressure)

z = np.polyfit(df5_withoutNA.Highest_Category, df5_withoutNA.Central_Pressure, 1)
p = np.poly1d(z)
plt.plot(df5_withoutNA.Highest_Category,p(df5_withoutNA.Highest_Category),"r--")

plt.title("Relationship between Highest Category and Central Pressure")
plt.xlabel("Highest Saffir-Simpson Category")
plt.ylabel("Central Pressure")
plt.show()

##Relationship between 'Highest_Category' and 'Max_Winds'
plt.scatter(df5_withoutNA.Highest_Category,df5_withoutNA.Max_Winds)

z = np.polyfit(df5_withoutNA.Highest_Category, df5_withoutNA.Max_Winds, 1)
p = np.poly1d(z)
plt.plot(df5_withoutNA.Highest_Category,p(df5_withoutNA.Highest_Category),"r--")

plt.title("Relationship between Highest Category and Max.Winds")
plt.xlabel("Highest Saffir-Simpson Category")
plt.ylabel("Central Pressure")
plt.show()

##Relationship between 'Central Pressure' and 'Max_Winds'
plt.scatter(df5_withoutNA.Max_Winds, df5_withoutNA.Central_Pressure)

z = np.polyfit(df5_withoutNA.Max_Winds, df5_withoutNA.Central_Pressure, 1)
p = np.poly1d(z)
plt.plot(df5_withoutNA.Max_Winds,p(df5_withoutNA.Max_Winds),"r--")

plt.title("Relationship between Max.Winds and Central Pressure")
plt.xlabel("Max.Winds")
plt.ylabel("Central Pressure")
plt.show()









