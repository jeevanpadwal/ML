import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
import statsmodels.api as sm
from itertools import cycle, islice

class CropAnalyzer:
    def __init__(self, df):
        """
        Initialize with preprocessed dataframe
        """
        self.df = df
        self.colors = plt.cm.tab10.colors
        
    def analyze_commodity_seasonality(self, commodity=None):
        """
        Analyze seasonality patterns for commodities
        """
        if commodity:
            # Filter for specific commodity
            commodity_df = self.df[self.df['Commodity'] == commodity].copy()
            commodities = [commodity]
        else:
            # Use all commodities
            commodity_df = self.df.copy()
            commodities = self.df['Commodity'].unique()
        
        # Create a figure for seasonal patterns
        fig, axes = plt.subplots(2, 1, figsize=(12, 12))
        
        # Monthly average prices
        monthly_prices = {}
        
        for i, commodity in enumerate(commodities):
            comm_data = commodity_df[commodity_df['Commodity'] == commodity]
            
            if len(comm_data) < 12:  # Skip if not enough data
                continue
                
            # Add year-month column
            comm_data['YearMonth'] = comm_data['Date'].dt.to_period('M')
            
            # Calculate monthly average prices
            monthly_avg = comm_data.groupby(['YearMonth', 'Month'])['Modal_Price'].mean().reset_index()
            monthly_avg = monthly_avg.sort_values('YearMonth')
            
            # Plot monthly time series
            color = self.colors[i % len(self.colors)]
            axes[0].plot(monthly_avg['YearMonth'].astype(str), monthly_avg['Modal_Price'], 
                       marker='o', linestyle='-', label=commodity, color=color)
            
            # Calculate average price by month (across all years)
            month_avg = comm_data.groupby('Month')['Modal_Price'].mean()
            monthly_prices[commodity] = month_avg
        
        # Set the monthly time series plot
        axes[0].set_title('Monthly Average Prices by Commodity')
        axes[0].set_xlabel('Year-Month')
        axes[0].set_ylabel('Average Price (₹)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='best')
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=90)
        
        # Plot monthly seasonality for all commodities
        for i, (commodity, month_prices) in enumerate(monthly_prices.items()):
            color = self.colors[i % len(self.colors)]
            axes[1].plot(month_prices.index, month_prices.values, 
                       marker='o', linestyle='-', label=commodity, color=color)
            
        axes[1].set_title('Seasonal Price Patterns by Month')
        axes[1].set_xlabel('Month')
        axes[1].set_ylabel('Average Price (₹)')
        axes[1].set_xticks(range(1, 13))
        axes[1].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc='best')
        
        plt.tight_layout()
        return fig, monthly_prices
    
    def decompose_time_series(self, commodity, market=None):
        """
        Decompose time series into trend, seasonal, and residual components
        """
        # Filter data
        commodity_df = self.df[self.df['Commodity'] == commodity].copy()
        
        if market:
            commodity_df = commodity_df[commodity_df['Market'] == market]
        
        if len(commodity_df) < 30:  # Not enough data
            return None
            
        # Sort by date and set as index
        commodity_df = commodity_df.sort_values('Date')
        ts_df = commodity_df.set_index('Date')['Modal_Price']
        
        # Resample to daily frequency, filling gaps
        ts_df = ts_df.resample('D').mean().fillna(method='ffill')
        
        # Apply STL decomposition
        decomposition = sm.tsa.seasonal_decompose(ts_df, model='additive', period=30)
        
        # Plot decomposition
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        
        # Original data
        axes[0].plot(ts_df.index, ts_df.values)
        axes[0].set_title(f'Price Time Series: {commodity}')
        axes[0].set_ylabel('Price (₹)')
        
        # Trend component
        axes[1].plot(decomposition.trend.index, decomposition.trend.values)
        axes[1].set_title('Trend')
        axes[1].set_ylabel('Price (₹)')
        
        # Seasonal component
        axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values)
        axes[2].set_title('Seasonality')
        axes[2].set_ylabel('Price (₹)')
        
        # Residual component
        axes[3].plot(decomposition.resid.dropna().index, decomposition.resid.dropna().values)
        axes[3].set_title('Residuals')
        axes[3].set_ylabel('Price (₹)')
        
        plt.tight_layout()
        return fig, decomposition
    
    def create_season_crop_matrix(self):
        """
        Create a matrix of seasons and crops with their average prices
        """
        if 'Season' not in self.df.columns:
            print("Season information not available in dataset")
            return None
        
        # Calculate average price by commodity and season
        season_crop_avg = self.df.groupby(['Season', 'Commodity'])['Modal_Price'].mean().reset_index()
        
        # Pivot the data for visualization
        pivot_df = season_crop_avg.pivot(index='Season', columns='Commodity', values='Modal_Price')
        
        # Create a heatmap
        plt.figure(figsize=(14, 8))
        sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.1f', linewidths=0.5)
        plt.title('Average Commodity Prices by Season')
        plt.tight_layout()
        
        return pivot_df
    
    def analyze_weather_price_correlation(self, commodity, weather_vars=None):
        """
        Analyze correlation between weather variables and commodity prices
        """
        if weather_vars is None:
            weather_vars = ['Temperature', 'Humidity', 'PRECTOTCORR', 'PS', 'WS2M']
        
        # Filter for the specified commodity
        commodity_df = self.df[self.df['Commodity'] == commodity].copy()
        
        # Ensure all required columns exist
        existing_vars = [var for var in weather_vars if var in commodity_df.columns]
        
        if not existing_vars:
            print("No weather variables found in data")
            return None
            
        # Calculate correlation
        corr_vars = existing_vars + ['Modal_Price']
        corr_matrix = commodity_df[corr_vars].corr()
        
        # Plot correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
        plt.title(f'Weather and Price Correlation for {commodity}')
        plt.tight_layout()
        
        # For each weather variable, also plot the direct relationship
        fig, axes = plt.subplots(len(existing_vars), 1, figsize=(12, 4*len(existing_vars)))
        
        for i, var in enumerate(existing_vars):
            # Sort by the weather variable
            plot_df = commodity_df.sort_values(var)
            
            if len(existing_vars) > 1:
                ax = axes[i]
            else:
                ax = axes
                
            ax.scatter(plot_df[var], plot_df['Modal_Price'], alpha=0.5)
            ax.set_xlabel(var)
            ax.set_ylabel('Price (₹)')
            ax.set_title(f'{var} vs {commodity} Price')
            
            # Add a smoothed line
            try:
                z = np.polyfit(plot_df[var], plot_df['Modal_Price'], 1)
                p = np.poly1d(z)
                ax.plot(plot_df[var], p(plot_df[var]), "r--", alpha=0.8)
            except:
                pass  # Skip if line fitting fails
                
        plt.tight_layout()
        
        return corr_matrix
        
    def visualize_best_crops(self, ranked_crops, predictions):
        """
        Visualize the predicted prices for the top ranked crops
        """
        # Take top 5 crops or less
        top_n = min(5, len(ranked_crops))
        top_crops = [crop for crop, _ in ranked_crops[:top_n]]
        
        # Create figure for price predictions
        plt.figure(figsize=(12, 8))
        
        for i, crop in enumerate(top_crops):
            if crop in predictions:
                pred_df = predictions[crop]
                plt.plot(pred_df['Date'], pred_df['Predicted_Price'], 
                        label=f"{crop}", linewidth=2, color=self.colors[i])
        
        plt.title('Predicted Prices for Top Crops')
        plt.xlabel('Date')
        plt.ylabel('Predicted Price (₹)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Create a bar chart for expected profit
        scores = [metrics['score'] for _, metrics in ranked_crops[:top_n]]
        trends = [metrics['price_trend_percent'] for _, metrics in ranked_crops[:top_n]]
        prices = [metrics['avg_predicted_price'] for _, metrics in ranked_crops[:top_n]]
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        
        # Score chart
        axes[0].bar(top_crops, scores, color=self.colors[:top_n])
        axes[0].set_title('Overall Score (Higher is Better)')
        axes[0].set_xlabel('Crop')
        axes[0].set_ylabel('Score')
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)
        
        # Price trend chart
        axes[1].bar(top_crops, trends, color=self.colors[:top_n])
        axes[1].set_title('Expected Price Trend (%)')
        axes[1].set_xlabel('Crop')
        axes[1].set_ylabel('Price Change (%)')
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45)
        
        # Average price chart
        axes[2].bar(top_crops, prices, color=self.colors[:top_n])
        axes[2].set_title('Average Predicted Price')
        axes[2].set_xlabel('Crop')
        axes[2].set_ylabel('Price (₹)')
        plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        return top_crops
        
    def create_crop_recommendation_table(self, ranked_crops, predictions, season=None):
        """
        Create a detailed recommendation table for farmers
        """
        # Take top crops
        top_n = min(10, len(ranked_crops))
        recommendation_data = []
        
        for crop, metrics in ranked_crops[:top_n]:
            if crop in predictions:
                pred_df = predictions[crop]
                
                # Get price at start and end of prediction period
                start_price = pred_df['Predicted_Price'].iloc[0]
                end_price = pred_df['Predicted_Price'].iloc[-1]
                
                # Calculate additional metrics
                price_change = end_price - start_price
                percent_change = (price_change / start_price) * 100 if start_price > 0 else 0
                
                # Find peak price and when it occurs
                peak_price = pred_df['Predicted_Price'].max()
                peak_date = pred_df.loc[pred_df['Predicted_Price'].idxmax(), 'Date']
                
                # Find minimum price and when it occurs
                min_price = pred_df['Predicted_Price'].min()
                min_date = pred_df.loc[pred_df['Predicted_Price'].idxmin(), 'Date']
                
                # Create recommendation
                if percent_change > 10:
                    recommendation = "Strong Buy - Expected significant price increase"
                elif percent_change > 5:
                    recommendation = "Buy - Expected moderate price increase"
                elif percent_change > 0:
                    recommendation = "Hold - Expected slight price increase"
                elif percent_change > -5:
                    recommendation = "Neutral - Price expected to remain stable"
                else:
                    recommendation = "Avoid - Expected price decrease"
                
                # Add season-specific advice if available
                if season:
                    recommendation += f" for {season} season"
                
                recommendation_data.append({
                    'Crop': crop,
                    'Current Price': f"₹{start_price:.2f}",
                    'Expected Future Price': f"₹{end_price:.2f}",
                    'Price Change': f"₹{price_change:.2f} ({percent_change:.1f}%)",
                    'Peak Price': f"₹{peak_price:.2f} on {peak_date.strftime('%d-%b-%Y')}",
                    'Min Price': f"₹{min_price:.2f} on {min_date.strftime('%d-%b-%Y')}",
                    'Prediction Accuracy': f"{metrics['prediction_accuracy']:.1f}%",
                    'Recommendation': recommendation
                })
        
        # Convert to DataFrame for easy viewing
        recommendation_df = pd.DataFrame(recommendation_data)
        
        return recommendation_df
    
    def generate_farmer_report(self, recommendation_df, top_crops, season=None):
        """
        Generate a comprehensive report for farmers
        """
        # Create report header
        if season:
            title = f"Crop Recommendation Report for {season} Season"
        else:
            title = "Crop Recommendation Report"
            
        report = [
            "=" * 80,
            f"{title:^80}",
            "=" * 80,
            "",
            "SUMMARY OF RECOMMENDATIONS:",
            "------------------------",
            ""
        ]
        
        # Add top recommendations
        for i, crop in enumerate(top_crops, 1):
            crop_data = recommendation_df[recommendation_df['Crop'] == crop].iloc[0]
            report.append(f"{i}. {crop}: {crop_data['Recommendation']}")
            report.append(f"   Current Price: {crop_data['Current Price']}, Expected: {crop_data['Expected Future Price']}")
            report.append(f"   Peak Price: {crop_data['Peak Price']}")
            report.append("")
        
        report.extend([
            "DETAILED ANALYSIS:",
            "----------------",
            ""
        ])
        
        # Add detailed recommendations
        for _, row in recommendation_df.iterrows():
            report.append(f"Crop: {row['Crop']}")
            report.append(f"Current Price: {row['Current Price']}")
            report.append(f"Expected Future Price: {row['Expected Future Price']}")
            report.append(f"Price Change: {row['Price Change']}")
            report.append(f"Peak Price: {row['Peak Price']}")
            report.append(f"Minimum Price: {row['Min Price']}")
            report.append(f"Prediction Accuracy: {row['Prediction Accuracy']}")
            report.append(f"Recommendation: {row['Recommendation']}")
            report.append("-" * 40)
            report.append("")
            
        report.extend([
            "MARKET INSIGHTS:",
            "---------------",
            "",
            "• Best time to sell most crops appears to be during market peaks as noted above.",
            "• Weather conditions in Pune are projected to remain suitable for the recommended crops.",
            "• Consider crop rotation and diversification to minimize risk.",
            "",
            "DISCLAIMER:",
            "-----------",
            "These recommendations are based on historical data analysis and predictive models.",
            "Actual market conditions may vary. Always consult with local agricultural experts",
            "and consider your specific farm conditions before making planting decisions.",
            "",
            "=" * 80
        ])
        
        return "\n".join(report)