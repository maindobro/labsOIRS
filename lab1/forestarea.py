import pandas
import matplotlib.pyplot as plt

plt.style.use('ggplot')

FILE_PATH = 'forestarea.xls'
df = pandas.read_excel(FILE_PATH)

df.drop(columns=['Country Code', 'Indicator Name', 'Indicator Code'], axis=1, inplace=True)
df.set_index('Country Name').loc['Suriname'].plot()
df.set_index('Country Name').loc['American Samoa'].plot()
df.set_index('Country Name').loc['Seychelles'].plot()
df.set_index('Country Name').loc['Micronesia, Fed. Sts.'].plot()
df.set_index('Country Name').loc['Gabon'].plot()
plt.legend()
plt.show()


df.set_index('Country Name').sum().plot()
plt.show()



