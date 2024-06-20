#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv('AmazonSaleReport.csv')


# In[2]:


df.head()


# In[3]:


df.shape


# In[4]:


df.columns


# In[5]:


df.info()


# In[6]:


# check missing values
missing_values = df.isnull().sum()


# In[7]:


missing_values


# In[8]:


df.dropna(axis=1)


# In[9]:


df.duplicated(['Order ID', 'Category', 'Size','ship-postal-code']).sum()
#df.duplicated(['Order ID']).sum()


# In[10]:


df.duplicated(['Order ID','Category','Status','ship-postal-code','Category','Size']).sum()


# In[11]:


df_d=df[df.duplicated(['Order ID','Category','Status','ship-postal-code','Category','Size'])]
df_d


# In[12]:


#df=df.drop_duplicates(keep='first')
df=df.drop_duplicates(subset=['Order ID','Category','Status','ship-postal-code','Category','Size'])#(keep='first')


# In[13]:


df.shape


# In[14]:


#aggreagate sales data by date to see overall trends
df_sales=((df.groupby('Date')['Amount']).sum())
df_sales


# In[15]:


df_sales_date=df['Date'].value_counts()
df_sales
#Category_count=df['Category'].value_counts()


# In[16]:


# Define a function to plot bar charts for categorical columns
import matplotlib.pyplot as plt
def plot_bar_chart(column_name):
    plt.figure(figsize=(50, 6))
    df['Date'].value_counts().plot(kind='bar')
    plt.title(f'Bar Chart of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

# Plotting bar charts for the identified categorical columns
categorical_columns = ['Date']
for column in categorical_columns:
    plot_bar_chart(column)


# In[17]:


top_10_dates = df_sales_date.nlargest(10)
top_10_dates


# In[18]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[19]:


df['Date'] = pd.to_datetime(df['Date'])

# Extract day name
df['DayName'] = df['Date'].dt.day_name()

# Group by 'Date' and sum 'Amount'
df_sales = df.groupby('Date')['Amount'].sum()

# Get the top 10 dates by sales amount
top_10_dates = df_sales.nlargest(10)

# Create a DataFrame for the result
result = top_10_dates.reset_index()

# Add day names to the result
result['DayName'] = result['Date'].dt.day_name()

print(result)


# # Find the ralation between days and sales

# Here we try to find the any specific days for sales . Here almost all days for sales is on

# In[20]:


df_sales.head()


# In[ ]:





# In[21]:


# visualize the df_sales in matplotlib and seaborn


# In[22]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[23]:


#set plots
sns.set(style='whitegrid')

#plot sales over date
plt.figure(figsize=(14,7))
sns.lineplot(data=df, x='Date',y='Amount')
plt.title('Sales preformance over date')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.show()


# In[ ]:





# In[24]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



# Convert 'Date' column to datetime
#df['Date'] = pd.to_datetime(df['Date'])
df['Date']=pd.to_datetime(df['Date'])

# Set 'Date' as the index
df.set_index('Date', inplace=True)

# Resample data by quarter and sum the sales amounts
df_quarterly_sales = df.resample('Q').sum()

# Reset index to use 'Date' as a column for plotting
df_quarterly_sales.reset_index(inplace=True)

# Plot the quarterly sales data
sns.set(style='whitegrid')
plt.figure(figsize=(20, 7))
sns.lineplot(data=df_quarterly_sales, x='Date', y='Amount')
plt.title('Quarterly Sales Performance')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.show()

see the quareirly sales trande is good and increases 
# In[ ]:





# In[25]:


df['Category'].nunique()


# In[26]:


df['Order ID'].nunique()


# In[27]:


df.columns


# In[ ]:





# In[28]:


categorical_columns = df.select_dtypes(include=['object', 'bool']).columns
categorical_value_counts = [{col: df[col].value_counts() for col in categorical_columns}]

#numerical_summary, categorical_value_counts


# In[29]:


print(categorical_value_counts)


# In[30]:


# data visualization


# In[31]:


#histograms or box plots for numerical columns eg. amount


# In[32]:


plt.figure(figsize=(10,5))
plt.hist(df["Amount"])
plt.show()


# In[33]:


Status_count = df['Status'].value_counts()
Status_count


# In[ ]:





# In[54]:


def plot_bar_chart(column_name):
    plt.figure(figsize=(10, 6))
    df[column_name].value_counts().plot(kind='bar')
    plt.title(f'Bar Chart of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Count')
    plt.xticks(rotation=80)
    plt.show()
categorical_columns = ['Status']
for column in categorical_columns:
    plot_bar_chart(column)


# In[34]:


Category_count=df['Category'].value_counts()
Category_count


# In[60]:


def plot_bar_chart(column_name):
    plt.figure(figsize=(10, 6))
    df[column_name].value_counts().plot(kind='bar')
    plt.title(f'Bar Chart of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Count')
    plt.xticks(rotation=80)
    plt.show()
categorical_columns = ['Category']
for column in categorical_columns:
    plot_bar_chart(column)


# In[84]:


Status_count = df['Size'].value_counts()
Status_count


# In[ ]:





# In[ ]:





# In[85]:



# Define a function to plot bar charts for categorical columns
def plot_bar_chart(column_name):
    plt.figure(figsize=(10, 6))
    df[column_name].value_counts().plot(kind='bar')
    plt.title(f'Bar Chart of {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

# Plotting bar charts for the identified categorical columns
categorical_columns = ['Fulfilment', 'Sales Channel', 'ship-service-level', 'Courier Status']
for column in categorical_columns:
    plot_bar_chart(column)


# In[ ]:





# In[30]:


import matplotlib.dates as mdates

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%m-%d-%y')
# Scatter plot for Date vs. Amount
plt.figure(figsize=(12, 6))
plt.scatter(df['Date'], df['Amount'])
plt.title('Scatter Plot of Date vs. Amount')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[31]:


# Line chart for Date vs. Amount (aggregated by Date)
date_amount = df.groupby('Date')['Amount'].sum().reset_index()
plt.figure(figsize=(12, 6))
plt.plot(date_amount['Date'], date_amount['Amount'])
plt.title('Line Chart of Date vs. Total Amount')
plt.xlabel('Date')
plt.ylabel('Total Amount')
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# In[32]:


import seaborn as sns
import matplotlib.pyplot as plt

# Cross-tabulation for Category vs. Status
crosstab_category_status = pd.crosstab(df['Category'], df['Status'])

# Plot heatmap for Category vs. Status
plt.figure(figsize=(12, 8))
sns.heatmap(crosstab_category_status, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Heatmap of Category vs. Status')
plt.xlabel('Status')
plt.ylabel('Category')
plt.show()


# In[ ]:





# In[33]:


# Cross-tabulation for Category vs. Fulfilment
crosstab_category_fulfilment = pd.crosstab(df['Category'], df['Fulfilment'])

# Plot heatmap for Category vs. Fulfilment
plt.figure(figsize=(12, 8))
sns.heatmap(crosstab_category_fulfilment, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Heatmap of Category vs. Fulfilment')
plt.xlabel('Fulfilment')
plt.ylabel('Category')
plt.show()


# In[87]:


numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
numerical_columns


# In[35]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(df[numerical_columns])
plt.show()


# In[36]:


import seaborn as sns
import matplotlib.pyplot as plt

# Set the plot size
plt.figure(figsize=(20, 8))

# Create a clustered bar chart
sns.countplot(x='Category',data=df,hue='Status')

# Set plot title and labels
plt.title('Clustered Bar Chart of Category vs. Status')
plt.xlabel('Category')
plt.ylabel('Count')

# Rotate x labels for better readability
plt.xticks(rotation=50)

# Show plot
plt.show()


# In[37]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create a FacetGrid for faceted plots
g = sns.FacetGrid(df, col='Status', col_wrap=4, height=4)

# Map a bar plot onto the grid
g.map(sns.countplot, 'Category', order=df['Category'].unique())

# Set plot title and labels for each subplot
g.set_titles('{col_name} Status')
g.set_axis_labels('Category', 'Count')

# Rotate x labels for better readability
for ax in g.axes.flatten():
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

# Adjust layout
g.fig.tight_layout(w_pad=1)

# Show plot
plt.show()


# In[39]:


# Plot the distribution of orders across different ship-city, ship-state, and ship-country

# Set the plot size
plt.figure(figsize=(14, 8))

# Plot the distribution for ship-city
plt.subplot(3, 1, 1)
sns.countplot(y='ship-city', data=df, order=df['ship-city'].value_counts().index)
plt.title('Distribution of Orders across Ship-City')
plt.xlabel('Count')
plt.ylabel('Ship-City')

# Plot the distribution for ship-state
plt.subplot(3, 1, 2)
sns.countplot(y='ship-state', data=df, order=df['ship-state'].value_counts().index)
plt.title('Distribution of Orders across Ship-State')
plt.xlabel('Count')
plt.ylabel('Ship-State')

# Plot the distribution for ship-country
plt.subplot(3, 1, 3)
sns.countplot(y='ship-country', data=df, order=df['ship-country'].value_counts().index)
plt.title('Distribution of Orders across Ship-Country')
plt.xlabel('Count')
plt.ylabel('Ship-Country')

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()


# In[42]:


# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)

# Extract year, month, and day from the 'Date' column
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day


# In[43]:


monthly_sales = df.groupby(['Year', 'Month']).agg({'Amount': 'sum'}).reset_index()

# Pivot the data for better visualization
monthly_sales_pivot = monthly_sales.pivot(index='Month', columns='Year', values='Amount')

# Plot the trends
import matplotlib.pyplot as plt

monthly_sales_pivot.plot(kind='bar', figsize=(12, 6))
plt.title('Monthly Sales Trends')
plt.xlabel('Month')
plt.ylabel('Total Amount')
plt.show()


# In[45]:


correlation_matrix = df.corr()

# Visualize correlations using a heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[46]:


sns.boxplot(data=df, x='Amount')
plt.title('Box Plot of Amount')
plt.show()


# In[47]:


from scipy.stats import zscore

df['Amount_zscore'] = zscore(df['Amount'])
outliers = df[df['Amount_zscore'].abs() > 3] # Z-score threshold for outliers


# In[48]:


aggregated_data = df.groupby(['Category', 'Status']).agg({'Amount': ['sum', 'mean'], 'Qty': 'mean'}).reset_index()
aggregated_data


# In[49]:


b2b_segment = df[df['B2B'] == True]
fulfilled_by_segment = df[df['fulfilled-by'] == 'Easy Ship']
# Analyze each segment separately


# In[50]:


status_changes_over_time = df.groupby(['Date', 'Status']).size().unstack(fill_value=0)

status_changes_over_time.plot(kind='line', figsize=(12, 6))
plt.title('Patterns in Status Changes Over Time')
plt.show()


# In[51]:


fulfillment_efficiency = df.groupby(['Fulfilment', 'Status']).size().unstack(fill_value=0)

fulfillment_efficiency.plot(kind='bar', stacked=True, figsize=(20, 6))
plt.title('Fulfillment Efficiency')
plt.show()


# In[99]:


top_states = df['ship-state'].value_counts()#.nlargest(30).index
top_states


# In[53]:


import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the data is loaded in the DataFrame 'df'
# df = pd.read_csv('path/to/AmazonSaleReport.csv')

# Get the top 10 states based on order count
top_10_states = df['ship-state'].value_counts().nlargest(30).index

# Plot the distribution for the top 10 states
plt.figure(figsize=(12, 8))
sns.countplot(y='ship-state', data=df, order=top_10_states)
plt.title('Distribution of Orders across Top 10 Ship-States')
plt.xlabel('Count')
plt.ylabel('Ship-State')
plt.show()


# In[ ]:





# # Sales overview

# You can clearly see in the above distribution the dates 4,5 and 6 are repeted many no of times and also generated good amount . Hence these are  most saleable dates .
# 
# Make sure stocks for all saleble categories is maintained properly for repeated dates.

# In[ ]:





# as per the figure of the status shipped-delivered-buyer is the less than the shipped . We will try to impove the shipped status and control the cancelled order.

# # Product Analysis:

# 
# T-shirts and shirts are more salebale product. Blazzer and trousers doing well we can focus here .Watches are less saleable items in all products.

# #In size M,L,XL is the top three size sold , after that XXl,S and 3Xl is sold so maintain the stocks of these sizes.

# In[ ]:





# # Fulfillment Analysis

# 
# In fulfillment amazon is doing well.
# 
# T-shirt and shirt are top  shipped product.
# 
# We will focus on cancelled products to imporve the revenew.
# 
# T-shirt and shirt are more cancelled product.

# In[ ]:





# # Geographical Analysis: 

# As per data MAHARASHTRA state is on the top in sales 
# and BENGALURU is top sales city.

# In[ ]:




