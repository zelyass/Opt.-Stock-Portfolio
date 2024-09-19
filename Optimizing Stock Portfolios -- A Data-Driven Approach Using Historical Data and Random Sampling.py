import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import random

#-------------------------------------------------------------------------
#          I will use yfinance to fetch live historical data
#-------------------------------------------------------------------------

def fetch_historical_data(tickers, period='5y'):
    data = yf.download(tickers, period=period)['Adj Close']
    return data

#-------------------------------------------------------------------------
#          Function to calculate portfolio risk
#-------------------------------------------------------------------------

def get_port_risk(weights, df):
    returns_df = df.pct_change(1, fill_method=None).dropna()
    vcv = returns_df.cov()
    
    var_p = np.dot(np.transpose(weights), np.dot(vcv, weights))
    sd_p = np.sqrt(var_p)
    sd_p_annual = sd_p * np.sqrt(250)
    
    return sd_p_annual

#-------------------------------------------------------------------------
#          Function to calculate portfolio return
#-------------------------------------------------------------------------

def get_port_expected_return(weights, df):
    returns_df = df.pct_change(1, fill_method=None).dropna()
    avg_daily_returns = returns_df.mean()
    port_return_daily = np.dot(weights, avg_daily_returns)
    port_return_annual = ((1 + port_return_daily) ** 250) - 1
    
    return port_return_annual

#-------------------------------------------------------------------------
#          Function to optimize our portfolio
#-------------------------------------------------------------------------

def optimize_portfolio(df):
    num_stocks = len(df.columns)
    init_weights = [1/num_stocks] * num_stocks
    bounds = tuple((0, 1) for _ in range(num_stocks))
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    
    # Minimize the portfolio risk
    results = minimize(fun=get_port_risk, x0=init_weights, args=(df,), bounds=bounds, constraints=constraints)
    return results

#-------------------------------------------------------------------------
#          Function to generate random samples
#-------------------------------------------------------------------------

def generate_random_portfolios(tickers, n=20, combination_size=17):
    random_combinations = []
    for _ in range(n):
        random_combinations.append(random.sample(tickers, combination_size))
    return random_combinations

#-------------------------------------------------------------------------
#          Analysis part
#-------------------------------------------------------------------------

def analyze_portfolios(combinations, historical_data):
    portfolio_metrics = []
    
    for combination in combinations:
        df_subset = historical_data[combination]
        
        # Optimize the portfolio
        results = optimize_portfolio(df_subset)
        
        # Calculate risk and return
        port_risk = get_port_risk(results.x, df_subset)
        port_return = get_port_expected_return(results.x, df_subset)
        return_risk_ratio = port_return / port_risk
        
        portfolio_metrics.append({
            'Combination': combination,
            'Risk': port_risk,
            'Return': port_return,
            'Return/Risk Ratio': return_risk_ratio,
            'Weights': results.x
        })
    
    # Sort portfolios by Return/Risk Ratio in descending order
    sorted_portfolios = sorted(portfolio_metrics, key=lambda x: x['Return/Risk Ratio'], reverse=True)
    return sorted_portfolios

#-------------------------------------------------------------------------
#          Stock Selection
#-------------------------------------------------------------------------

tickers = [
    'SPY', 'QQQ', 'AMZN', 'AAPL', 'HES', 'KO', 'MRK', 'BSX', 
    'CP.TO', 'TFII.TO', 'TVK.TO', 'BAC', 'AVGO', 'ISRG', 'DIS', 
    'PFE', 'MSFT', 'TMO', 'T', 'HUM', 'SPGI', 'LLY', 'EQIX', 
    'ICLR', 'TRU', 'PYPL', 'TSM', 'TSLA', 'CSCO', 'BRK-B'
]

df = fetch_historical_data(tickers)

# Generate random portfolios
random_combinations = generate_random_portfolios(tickers, n=30, combination_size=17)

# Analyze the portfolios and get the top five
portfolio_analysis = analyze_portfolios(random_combinations, df)
top_five_portfolios = portfolio_analysis[:5]

#-------------------------------------------------------------------------
#          Displaying top Five Portfolios
#-------------------------------------------------------------------------

def display_portfolios(portfolios):
    portfolio_dfs = []
    
    for portfolio in portfolios:
        df = pd.DataFrame({
            'Ticker': portfolio['Combination'],
            'Weight': portfolio['Weights']
        })
        
        # Add return, risk, and ratio as separate lines
        summary = pd.DataFrame({
            'Metric': ['Return', 'Risk', 'Return/Risk Ratio'],
            'Value': [portfolio['Return'], portfolio['Risk'], portfolio['Return/Risk Ratio']]
        }, index=['Return', 'Risk', 'Return/Risk Ratio'])
        
        # Combine the ticker weights with the summary
        portfolio_summary = pd.concat([df, summary], axis=1)
        portfolio_dfs.append(portfolio_summary)
    
    return portfolio_dfs

# Display top five portfolios

top_five_portfolio_dfs = display_portfolios(top_five_portfolios)
for i, portfolio_df in enumerate(top_five_portfolio_dfs, 1):
  print(f"\nTop Portfolio {i}")
  print(portfolio_df)
  
View(top_five_portfolio_dfs)
