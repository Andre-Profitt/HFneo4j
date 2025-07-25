#!/usr/bin/env python3
"""
Standalone demo of the Hedge Fund Graph Stack
Demonstrates core algorithms without requiring Neo4j/Docker
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import networkx as nx
from sklearn.cluster import SpectralClustering
from scipy.linalg import sqrtm
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

print("🚀 Hedge Fund Graph Stack - Standalone Demo\n")

# 1. SIMULATE MARKET DATA
print("1️⃣ Generating synthetic market data...")
np.random.seed(42)
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'BAC', 'GS', 'MS',
           'XOM', 'CVX', 'PFE', 'JNJ', 'UNH', 'WMT', 'HD', 'DIS', 'NFLX', 'TSLA']
n_days = 252
dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')

# Generate correlated returns
n_stocks = len(symbols)
# Create correlation structure (tech stocks correlated, financials correlated, etc)
base_corr = np.eye(n_stocks) 
# Tech cluster (0-5)
base_corr[0:6, 0:6] += np.random.rand(6, 6) * 0.3
# Financial cluster (6-9)
base_corr[6:10, 6:10] += np.random.rand(4, 4) * 0.4
# Healthcare cluster (12-14)
base_corr[12:15, 12:15] += np.random.rand(3, 3) * 0.35

# Make symmetric and ensure valid correlation matrix
base_corr = (base_corr + base_corr.T) / 2
np.fill_diagonal(base_corr, 1)
# Ensure positive semi-definite
eigenvalues, eigenvectors = np.linalg.eigh(base_corr)
eigenvalues[eigenvalues < 0] = 0.01
base_corr = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

# Generate returns
mean_returns = np.random.randn(n_stocks) * 0.0002 + 0.0003  # Daily returns
volatilities = np.random.uniform(0.01, 0.03, n_stocks)  # Daily volatility
cov_matrix = np.outer(volatilities, volatilities) * base_corr

returns_data = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)
returns = pd.DataFrame(returns_data, index=dates, columns=symbols)
prices = (1 + returns).cumprod() * 100  # Start at $100

print(f"✅ Generated {n_days} days of returns for {n_stocks} stocks")
print(f"   Average daily return: {returns.mean().mean():.2%}")
print(f"   Average volatility: {returns.std().mean():.2%}")

# 2. GRAPH-ENHANCED FACTOR MODEL (GEFM)
print("\n2️⃣ Running Graph-Enhanced Factor Model (GEFM)...")

# Calculate correlation matrix
corr_matrix = returns.corr()
print(f"   Correlation matrix shape: {corr_matrix.shape}")

# Build graph from correlations
threshold = 0.4
adj_matrix = (np.abs(corr_matrix.values) > threshold).astype(float)
np.fill_diagonal(adj_matrix, 0)  # No self-loops

# Spectral clustering
n_clusters = 5
clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
cluster_labels = clustering.fit_predict(adj_matrix + np.eye(n_stocks) * 0.1)

# Create factor loadings
loadings = pd.DataFrame(0, index=symbols, columns=[f'Factor_{i}' for i in range(n_clusters)])
for i, symbol in enumerate(symbols):
    loadings.loc[symbol, f'Factor_{cluster_labels[i]}'] = 1

# Calculate factor returns
factor_returns = pd.DataFrame(index=dates, columns=loadings.columns)
for factor in loadings.columns:
    stocks_in_factor = loadings[loadings[factor] == 1].index
    factor_returns[factor] = returns[stocks_in_factor].mean(axis=1)

# Risk decomposition
factor_cov = factor_returns.cov()
specific_returns = returns - (loadings @ factor_returns.T).T
specific_risk = specific_returns.std()
total_risk = returns.std()
risk_reduction = 1 - (specific_risk / total_risk).mean()

print(f"✅ GEFM Results:")
print(f"   Clusters found: {n_clusters}")
print(f"   Risk reduction: {risk_reduction:.1%}")
print(f"   Factor correlations:")
print(factor_cov.round(3))

# Visualize clusters
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
cluster_df = pd.DataFrame({'Symbol': symbols, 'Cluster': cluster_labels})
cluster_df['Cluster'].value_counts().sort_index().plot(kind='bar')
plt.title('GEFM: Cluster Sizes')
plt.xlabel('Cluster ID')
plt.ylabel('Number of Stocks')

plt.subplot(1, 2, 2)
# Plot correlation heatmap with clusters
sorted_idx = np.argsort(cluster_labels)
sorted_corr = corr_matrix.iloc[sorted_idx, sorted_idx]
sns.heatmap(sorted_corr, cmap='RdBu_r', center=0, vmin=-1, vmax=1, 
            square=True, cbar_kws={'label': 'Correlation'})
plt.title('GEFM: Correlation Matrix (Clustered)')
plt.tight_layout()
plt.savefig('gefm_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved visualization to gefm_results.png")

# 3. DEBTRANK CONTAGION ANALYSIS
print("\n3️⃣ Running DebtRank Contagion Analysis...")

# Create synthetic exposure network
entities = ['HF_Alpha', 'HF_Beta', 'HF_Gamma', 'PB_Goldman', 'PB_Morgan', 'CCP_CME']
n_entities = len(entities)

# Exposure matrix (who owes whom)
exposure_matrix = np.array([
    [0,    0,    0,    1500, 800,  0],     # HF_Alpha
    [0,    0,    0,    2000, 1500, 0],     # HF_Beta  
    [0,    0,    0,    0,    1000, 0],     # HF_Gamma
    [0,    0,    0,    0,    0,    5000],  # PB_Goldman
    [0,    0,    0,    0,    0,    4500],  # PB_Morgan
    [0,    0,    0,    0,    0,    0]      # CCP_CME
]) * 1e6  # Millions

# Normalize by total assets
total_assets = np.array([5000, 8000, 3000, 50000, 45000, 100000]) * 1e6
normalized_exposure = exposure_matrix / total_assets[:, np.newaxis]

# DebtRank algorithm
def calculate_debtrank(exposure_norm, initial_shock, max_iter=100, tol=1e-6):
    n = len(initial_shock)
    h = initial_shock.copy()
    h_history = [h.copy()]
    
    for i in range(max_iter):
        h_new = h + exposure_norm.T @ h
        h_new = np.minimum(h_new, 1.0)  # Cap at 1
        
        if np.max(np.abs(h_new - h)) < tol:
            break
            
        h = h_new
        h_history.append(h.copy())
        
    return h, np.array(h_history)

# Simulate shock to HF_Alpha (15% loss)
initial_shock = np.zeros(n_entities)
initial_shock[0] = 0.15

h_final, h_history = calculate_debtrank(normalized_exposure, initial_shock)

print(f"✅ DebtRank Results:")
print(f"   Initial shock: {entities[0]} = {initial_shock[0]:.1%}")
print(f"   Contagion impact:")
for i, entity in enumerate(entities):
    if h_final[i] > 0.01:
        print(f"     {entity}: {h_final[i]:.1%}")
print(f"   Total systemic impact: {h_final.sum():.2f}")
print(f"   Amplification factor: {h_final.sum()/initial_shock.sum():.2f}x")

# Visualize contagion
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
colors = ['red' if i == 0 else 'orange' if h_final[i] > 0.05 else 'green' for i in range(n_entities)]
plt.bar(entities, h_final, color=colors)
plt.title('DebtRank: Contagion Impact')
plt.ylabel('Impact Score')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
# Create network visualization
G = nx.DiGraph()
for i in range(n_entities):
    G.add_node(entities[i], impact=h_final[i])
for i in range(n_entities):
    for j in range(n_entities):
        if exposure_matrix[i, j] > 0:
            G.add_edge(entities[i], entities[j], weight=exposure_matrix[i, j]/1e9)

pos = nx.spring_layout(G, k=2, iterations=50)
node_colors = [h_final[entities.index(node)] for node in G.nodes()]
node_sizes = [3000 * (1 + h_final[entities.index(node)]) for node in G.nodes()]

nx.draw(G, pos, node_color=node_colors, node_size=node_sizes, 
        cmap='Reds', vmin=0, vmax=0.5, with_labels=True, 
        arrows=True, edge_color='gray', alpha=0.7)
plt.title('DebtRank: Exposure Network')
plt.tight_layout()
plt.savefig('gefm_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved visualization to gefm_results.png")

# 4. GRAPH-SIGNAL RISK PARITY
print("\n4️⃣ Running Graph-Signal Risk Parity (GSRP)...")

# Construct Laplacian from correlation graph
W = (np.abs(corr_matrix) > 0.3).astype(float) * np.abs(corr_matrix)
np.fill_diagonal(W, 0)
D = np.diag(W.sum(axis=1))
L = D - W  # Laplacian

# Normalized Laplacian
D_sqrt_inv = np.diag(1.0 / np.sqrt(np.maximum(D.diagonal(), 1e-8)))
L_norm = D_sqrt_inv @ L @ D_sqrt_inv

# Standard risk parity weights (equal risk contribution)
cov = returns.cov()
inv_vol = 1 / returns.std()
rp_weights = inv_vol / inv_vol.sum()

# Graph-regularized weights (simplified version)
# Minimize: w'*Cov*w + lambda*w'*L*w
lambda_graph = 0.5
regularized_cov = cov.values + lambda_graph * L_norm
try:
    inv_reg_cov = np.linalg.inv(regularized_cov + np.eye(n_stocks) * 1e-6)
    ones = np.ones(n_stocks)
    raw_weights = inv_reg_cov @ ones
    gsrp_weights = raw_weights / raw_weights.sum()
    gsrp_weights = pd.Series(gsrp_weights, index=symbols)
except:
    gsrp_weights = rp_weights.copy()

# Calculate metrics
rp_vol = np.sqrt(rp_weights @ cov @ rp_weights)
gsrp_vol = np.sqrt(gsrp_weights @ cov @ gsrp_weights)
rp_gtv = rp_weights.values @ L_norm @ rp_weights.values
gsrp_gtv = gsrp_weights.values @ L_norm @ gsrp_weights.values

print(f"✅ GSRP Results:")
print(f"   Standard RP volatility: {rp_vol*np.sqrt(252):.2%} annual")
print(f"   GSRP volatility: {gsrp_vol*np.sqrt(252):.2%} annual")
print(f"   Volatility reduction: {(rp_vol-gsrp_vol)/rp_vol:.1%}")
print(f"   Graph Total Variation reduction: {(rp_gtv-gsrp_gtv)/rp_gtv:.1%}")

# Visualize weights
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
weights_df = pd.DataFrame({
    'Risk Parity': rp_weights,
    'Graph-Signal RP': gsrp_weights
})
weights_df.plot(kind='bar', width=0.8)
plt.title('Portfolio Weights Comparison')
plt.ylabel('Weight')
plt.xticks(rotation=45)
plt.legend()

plt.subplot(1, 2, 2)
# Show weight differences colored by cluster
weight_diff = gsrp_weights - rp_weights
colors = [plt.cm.tab10(cluster_labels[i]) for i in range(n_stocks)]
plt.bar(symbols, weight_diff, color=colors)
plt.title('GSRP vs RP Weight Differences (Colored by Cluster)')
plt.ylabel('Weight Difference')
plt.xticks(rotation=45)
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.savefig('gefm_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved visualization to gefm_results.png")

# 5. PERFORMANCE BACKTESTING
print("\n5️⃣ Running Performance Backtest...")

# Calculate portfolio returns
rp_returns = (returns * rp_weights).sum(axis=1)
gsrp_returns = (returns * gsrp_weights).sum(axis=1)
equal_returns = returns.mean(axis=1)

# Calculate cumulative returns
cum_rp = (1 + rp_returns).cumprod()
cum_gsrp = (1 + gsrp_returns).cumprod()
cum_equal = (1 + equal_returns).cumprod()

# Performance metrics
def calc_metrics(returns_series):
    total_ret = (1 + returns_series).prod() - 1
    annual_ret = (1 + total_ret) ** (252/len(returns_series)) - 1
    vol = returns_series.std() * np.sqrt(252)
    sharpe = annual_ret / vol
    
    # Max drawdown
    cum_ret = (1 + returns_series).cumprod()
    running_max = cum_ret.expanding().max()
    drawdown = (cum_ret - running_max) / running_max
    max_dd = drawdown.min()
    
    return {
        'Annual Return': annual_ret,
        'Volatility': vol,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_dd
    }

metrics_df = pd.DataFrame({
    'Equal Weight': calc_metrics(equal_returns),
    'Risk Parity': calc_metrics(rp_returns),
    'Graph-Signal RP': calc_metrics(gsrp_returns)
}).T

print("✅ Backtest Results:")
print(metrics_df.round(3))

# Visualize performance
plt.figure(figsize=(14, 8))

plt.subplot(2, 2, 1)
cum_equal.plot(label='Equal Weight', alpha=0.7)
cum_rp.plot(label='Risk Parity', alpha=0.7)
cum_gsrp.plot(label='Graph-Signal RP', alpha=0.7)
plt.title('Cumulative Returns')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
# Rolling volatility
rolling_vol = pd.DataFrame({
    'Equal Weight': equal_returns.rolling(60).std() * np.sqrt(252),
    'Risk Parity': rp_returns.rolling(60).std() * np.sqrt(252),
    'Graph-Signal RP': gsrp_returns.rolling(60).std() * np.sqrt(252)
})
rolling_vol.plot(alpha=0.7)
plt.title('Rolling 60-Day Volatility')
plt.ylabel('Annualized Volatility')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
# Drawdowns
for series, name in [(equal_returns, 'Equal Weight'), 
                     (rp_returns, 'Risk Parity'),
                     (gsrp_returns, 'Graph-Signal RP')]:
    cum_ret = (1 + series).cumprod()
    running_max = cum_ret.expanding().max()
    dd = (cum_ret - running_max) / running_max
    dd.plot(label=name, alpha=0.7)
plt.title('Drawdowns')
plt.ylabel('Drawdown %')
plt.legend()
plt.grid(True, alpha=0.3)
plt.fill_between(dates, 0, -0.2, alpha=0.1, color='red')

plt.subplot(2, 2, 4)
# Performance metrics bar chart
metrics_df[['Sharpe Ratio', 'Annual Return']].plot(kind='bar', width=0.8)
plt.title('Performance Metrics Comparison')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gefm_results.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved visualization to gefm_results.png")

# 6. SUMMARY STATISTICS
print("\n📊 FINAL SUMMARY")
print("="*50)
print(f"Graph-Enhanced Factor Model:")
print(f"  - Specific risk reduction: {risk_reduction:.1%}")
print(f"  - Number of factors: {n_clusters}")
print(f"\nDebtRank Contagion:")
print(f"  - Systemic amplification: {h_final.sum()/initial_shock.sum():.1f}x")
print(f"  - Entities affected: {(h_final > 0.01).sum()}/{n_entities}")
print(f"\nGraph-Signal Risk Parity:")
print(f"  - Volatility vs Equal Weight: {(metrics_df.loc['Graph-Signal RP', 'Volatility'] / metrics_df.loc['Equal Weight', 'Volatility'] - 1)*100:.1f}%")
print(f"  - Sharpe vs Risk Parity: {(metrics_df.loc['Graph-Signal RP', 'Sharpe Ratio'] / metrics_df.loc['Risk Parity', 'Sharpe Ratio'] - 1)*100:+.1f}%")
print(f"\nBacktest Performance (Graph-Signal RP):")
print(f"  - Annual Return: {metrics_df.loc['Graph-Signal RP', 'Annual Return']:.1%}")
print(f"  - Sharpe Ratio: {metrics_df.loc['Graph-Signal RP', 'Sharpe Ratio']:.2f}")
print(f"  - Max Drawdown: {metrics_df.loc['Graph-Signal RP', 'Max Drawdown']:.1%}")
print("="*50)

print("\n✅ Demo completed! This demonstrates the core algorithms of the")
print("   Hedge Fund Graph Stack without requiring Neo4j/Docker infrastructure.")
print("   In production, these algorithms would operate on real-time data")
print("   with Neo4j providing the graph database backend.")