import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
stock_yfinance_data = pd.read_csv('data/stock_yfinance_data.csv')
stock_tweets = pd.read_csv('data/stock_tweets.csv')
reduced_dataset_release = pd.read_csv('data/reduced_dataset-release.csv')

# Normalize the date format for consistency and remove time components
stock_yfinance_data['Date'] = pd.to_datetime(stock_yfinance_data['Date']).dt.date
stock_tweets['Date'] = pd.to_datetime(stock_tweets['Date'], errors='coerce').dt.date

# Rename columns in reduced_dataset_release to match the other datasets
reduced_dataset_release.rename(columns={
    'DATE': 'Date',
    'STOCK': 'Stock Name',
    'LAST_PRICE': 'Close',
    'TWEET': 'Tweet'
}, inplace=True)

# Ensure the `Date` column in reduced_dataset_release is properly formatted
reduced_dataset_release['Date'] = pd.to_datetime(reduced_dataset_release['Date'], errors='coerce').dt.date

# Select necessary columns for merging
stock_yfinance_subset = stock_yfinance_data[['Date', 'Stock Name', 'Close']]
stock_tweets_subset = stock_tweets[['Date', 'Stock Name', 'Tweet']]
reduced_dataset_release_subset = reduced_dataset_release[['Date', 'Stock Name', 'Close', 'Tweet', 'LSTM_POLARITY']]

# Merge the first two datasets (stock prices and tweets)
optimized_merge = pd.merge(stock_yfinance_subset, stock_tweets_subset, on=['Date', 'Stock Name'], how='outer')

# Merge the result with the third dataset to include LSTM_POLARITY and tweets from the reduced dataset
final_combined_data = pd.merge(optimized_merge, reduced_dataset_release_subset, on=['Date', 'Stock Name', 'Close', 'Tweet'], how='outer')

# Clean the dataset
#The third dataset has a lot of nonsense / misaligned values which I will need to clean here:


# Fix Stock Name column
# If `Stock Name` doesn't match the pattern, mark them as invalid (This first step was done using the aid of CHATGPT - It is the only step in this process) 
valid_stock_pattern = r'^[A-Z0-9]+$'
final_combined_data['Stock Name'] = final_combined_data['Stock Name'].where(
    final_combined_data['Stock Name'].str.match(valid_stock_pattern, na=False), None
)

# Fix Date column
# Ensure `Date` contains only valid dates
final_combined_data['Date'] = pd.to_datetime(final_combined_data['Date'], errors='coerce').dt.date

# Fix Close column
# Remove nonsense values like negatives or extremely small numbers (less than 0.01)
final_combined_data['Close'] = pd.to_numeric(final_combined_data['Close'], errors='coerce')
final_combined_data.loc[(final_combined_data['Close'] < 0.01) | (final_combined_data['Close'].isna()), 'Close'] = None

# Drop rows where critical columns (Date, Stock Name, or Close) are missing after cleanup
cleaned_data = final_combined_data.dropna(subset=['Date', 'Stock Name', 'Close'])

#Handling Duplicates and Filling missing values
cleaned_data_no_duplicates = cleaned_data.drop_duplicates()

# Step 2: Fill missing values
# Fill missing values for 'LSTM_POLARITY' with 0 (neutral sentiment)
cleaned_data_no_duplicates['LSTM_POLARITY'] = cleaned_data_no_duplicates['LSTM_POLARITY'].fillna(0)

# Fill missing values in 'Tweet' with a placeholder string
cleaned_data_no_duplicates['Tweet'] = cleaned_data_no_duplicates['Tweet'].fillna('NaN')
cleaned_data_no_duplicates.head()

#Filtering and Sorting the Data
cleaned_data_no_duplicates.loc[:, 'LSTM_POLARITY'] = pd.to_numeric(
    cleaned_data_no_duplicates['LSTM_POLARITY'], errors='coerce'
)

# Fix 2: Convert `Date` to datetime consistently
cleaned_data_no_duplicates.loc[:, 'Date'] = pd.to_datetime(
    cleaned_data_no_duplicates['Date'], errors='coerce'
)

# Filtering Subsets of Data
# Example 1: Filter for a specific stock, e.g., Tesla (TSLA)
filtered_tesla = cleaned_data_no_duplicates[cleaned_data_no_duplicates['Stock Name'] == 'TSLA']

# Example 2: Filter by sentiment polarity (e.g., positive sentiment)
filtered_positive_sentiment = cleaned_data_no_duplicates[cleaned_data_no_duplicates['LSTM_POLARITY'] > 0]

# Example 3: Filter by a date range (e.g., data from 2021-01-01 to 2021-12-31)
filtered_date_range = cleaned_data_no_duplicates[
    (cleaned_data_no_duplicates['Date'] >= pd.Timestamp('2021-01-01')) &
    (cleaned_data_no_duplicates['Date'] <= pd.Timestamp('2021-12-31'))
]

# Sorting the Data
# Sort by Date to observe trends over time
sorted_by_date = cleaned_data_no_duplicates.sort_values(by='Date')

# Sort by Close to observe highest and lowest stock prices
sorted_by_close = cleaned_data_no_duplicates.sort_values(by='Close', ascending=False)

# Sort by LSTM_POLARITY to observe the most positive and negative sentiment
sorted_by_sentiment = cleaned_data_no_duplicates.sort_values(by='LSTM_POLARITY', ascending=False)

# Visualization 1: Boxplot - Variability in stock prices for different stocks
# Selected top 5 stocks with the most data
top_stocks = cleaned_data_no_duplicates['Stock Name'].value_counts().head(5).index
boxplot_data = cleaned_data_no_duplicates[cleaned_data_no_duplicates['Stock Name'].isin(top_stocks)]

plt.figure(figsize=(12, 6))
boxplot_data.boxplot(column='Close', by='Stock Name', grid=False, showfliers=False, patch_artist=True)
plt.title('Variability in Stock Prices Across Top 5 Stocks')
plt.xlabel('Stock Name')
plt.ylabel('Close Price')
plt.show()

# Visualization 2: Line Chart - Trends in Close prices over time for selected stocks
# Filter data for Tesla (TSLA) and Apple (AAPL)
selected_stocks = cleaned_data_no_duplicates[cleaned_data_no_duplicates['Stock Name'].isin(['TSLA', 'AAPL'])]

plt.figure(figsize=(12, 6))
for stock in selected_stocks['Stock Name'].unique():
    stock_data = selected_stocks[selected_stocks['Stock Name'] == stock]
    plt.plot(stock_data['Date'], stock_data['Close'], label=stock)

plt.title('Trends in Close Prices Over Time (TSLA and AAPL)')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend(title='Stock Name')
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.show()

# Visualization 3: Pie Chart - Stock Mentions Distribution (Top 10 Stocks and Others)
# Count mentions of stocks
stock_mentions = cleaned_data_no_duplicates['Stock Name'].value_counts()

# Separate top 10 stocks and group the rest as "Other"
top_10_stocks = stock_mentions.nlargest(10)
others = stock_mentions.iloc[10:].sum()

# Prepare data for pie chart
stock_mentions_pie = pd.concat([top_10_stocks, pd.Series({'Other': others})])
stock_mentions_percentage = (stock_mentions_pie / stock_mentions_pie.sum()) * 100

plt.figure(figsize=(10, 8))
plt.pie(stock_mentions_percentage, labels=stock_mentions_pie.index, autopct='%1.1f%%', startangle=90)
plt.title('Stock Mentions Distribution (Top 10 Stocks and Others)')
plt.show()

# Visualization 4: Scatter Plot for Tweet Counts by Stock
# Group by Stock Name and Date to get tweet counts
tweet_counts = cleaned_data_no_duplicates.groupby(['Stock Name', 'Date']).size().reset_index(name='Tweet Count')

# Get unique stock names for iteration
unique_stocks = tweet_counts['Stock Name'].unique()

# Create subplots for each stock
fig, axs = plt.subplots(5, 5, figsize=(15, 15))  # 5x5 grid for top 25 stocks
axs = axs.flatten()

# Plot tweet trends for each stock
for i, stock in enumerate(unique_stocks[:25]):  # Limiting to top 25 stocks for clarity
    stock_data = tweet_counts[tweet_counts['Stock Name'] == stock]
    axs[i].scatter(stock_data['Date'], stock_data['Tweet Count'], alpha=0.6, color='orange')
    axs[i].set_title(stock)
    axs[i].set_xlabel('Date')
    axs[i].set_ylabel('Tweet Count')
    axs[i].tick_params(axis='x', rotation=45)

# Adjust layout and show the plots
plt.tight_layout()
plt.suptitle('Trend for Tweet Counts by Stock', y=1.02, fontsize=16)
plt.show()

#Visualization 5
# Filter data for Tesla (TSLA)
tesla_data = cleaned_data_no_duplicates[cleaned_data_no_duplicates['Stock Name'] == 'TSLA']

# Group by Date to calculate daily tweet counts and average close price
tesla_trend = tesla_data.groupby('Date').agg(
    Tweet_Count=('Tweet', 'count'),  # Count the number of tweets
    Average_Close=('Close', 'mean')  # Calculate the average close price
).reset_index()

# Plot dual-axis line chart
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot stock price on the first y-axis
ax1.set_xlabel('Date')
ax1.set_ylabel('Average Close Price', color='tab:blue')
ax1.plot(tesla_trend['Date'], tesla_trend['Average_Close'], label='Average Close Price', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Plot tweet counts on the second y-axis
ax2 = ax1.twinx()
ax2.set_ylabel('Tweet Count', color='tab:orange')
ax2.plot(tesla_trend['Date'], tesla_trend['Tweet_Count'], label='Tweet Count', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

# Add title and legends
fig.suptitle('Tesla Mention Count vs. Stock Price Trend for TESLA')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()

#Pivoting
# Select top 5 stocks for pivoting
top_5_stocks = cleaned_data_no_duplicates['Stock Name'].value_counts().head(5).index

# Filter the data for top 5 stocks
pivot_data = cleaned_data_no_duplicates[cleaned_data_no_duplicates['Stock Name'].isin(top_5_stocks)]

# Pivot the data: rows = Date, columns = Stock Name, values = Close
pivot_table = pivot_data.pivot_table(
    index='Date', columns='Stock Name', values='Close', aggfunc='mean'
)

pivot_table.head()

#Stacking
stack_data = cleaned_data_no_duplicates[['Date', 'Stock Name', 'Close', 'Tweet', 'LSTM_POLARITY']]

# Stack the data: set `Date` and `Stock Name` as the index, stack the remaining columns
stacked_data = stack_data.set_index(['Date', 'Stock Name']).stack()
stacked_data.head()
