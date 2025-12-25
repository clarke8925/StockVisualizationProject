# StockVisualizationProject
The primary objective of this project is to analyze the relationship between stock prices and social media activity using real-world datasets. By combining historical stock prices, sentiment data from social media posts, and aggregated metrics, we aim to uncover trends and correlations that could shed light on market behaviors.
Our motivation stems from the increasing influence of social media sentiment on financial markets. This project demonstrates data wrangling, filtering, and visualization techniques to explore the intricate dynamics between these variables.

2. Datasets 
We selected the following datasets for analysis: 

1. Stock Prices Dataset (Yahoo Finance)
   - Source: Yahoo Finance API 
   - Size: 63,000 Rows 
   - Attributes: Date, Stock Name, Close Price

2. Social Media Tweets Dataset
   - Source: Twitter API (aggregated by Kaggle)
   - Size: 80,793 Rows 
   - Attributes: Date, Stock Name, Tweet Content

3. Reduced Aggregated Sentiment Dataset
   - Source: Internal aggregation (released dataset)
   - Size: 182,864 Rows 
   - Attributes: Date, Stock Name, Close Price, LSTM Polarity, Tweets
