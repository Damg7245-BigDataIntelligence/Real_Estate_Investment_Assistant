import google.generativeai as genai
from dotenv import load_dotenv
import os
import io
import sys
import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
            
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from backend.s3_utils import upload_visualization_to_s3

load_dotenv()

class GeminiVisualizationAgent:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        self.vision_model = genai.GenerativeModel('gemini-1.5-flash')

    def _extract_first_number(self, value: str) -> float:
        """Extract first number from string."""
        try:
            if isinstance(value, (int, float)):
                return float(value)
            
            # Remove currency symbols and commas
            cleaned = str(value).replace('$', '').replace(',', '')
            
            # Try to find first number (including decimals)
            matches = re.findall(r'(\d+\.?\d*)', cleaned)
            if matches:
                return float(matches[0])
            
            # If no matches, try to extract just first digit
            digit_match = re.search(r'\d', cleaned)
            if digit_match:
                return float(digit_match.group())
            
            return 0.0
            
        except (ValueError, AttributeError):
            return 0.0

    def _serialize_date(self, obj):
        """Custom JSON serializer for handling dates."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)

    def _calculate_listing_stats(self, listings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistical information about the listings."""
        if not listings:
            return {}

        try:
            print("Calculating listing stats...")
            prices = []
            beds = []
            baths = []
            sqft = []
            cities = {}
            price_per_sqft = []
            zip_codes = {}

            for listing in listings:
                # Handle Price
                price_val = 0
                if 'Price' in listing and listing['Price']:
                    price_val = self._extract_first_number(listing['Price'])
                    if price_val > 0:
                        prices.append(price_val)

                # Handle Beds
                bed_val = 0
                if 'Beds' in listing and listing['Beds']:
                    bed_val = self._extract_first_number(str(listing['Beds']))
                    if bed_val > 0:
                        beds.append(bed_val)

                # Handle Baths
                bath_val = 0
                if 'Baths' in listing and listing['Baths']:
                    bath_val = self._extract_first_number(str(listing['Baths']))
                    if bath_val > 0:
                        baths.append(bath_val)

                # Handle Sqft
                sqft_val = 0
                if 'Sqft' in listing and listing['Sqft']:
                    sqft_val = self._extract_first_number(str(listing['Sqft']))
                    if sqft_val > 0:
                        sqft.append(sqft_val)
                
                # Calculate price per sqft
                if price_val > 0 and sqft_val > 0:
                    price_per_sqft.append(price_val / sqft_val)
                
                # Track cities
                if 'City' in listing and listing['City']:
                    city = listing['City']
                    cities[city] = cities.get(city, 0) + 1
                
                # Track zip codes
                if 'ZipCode' in listing and listing['ZipCode']:
                    zip_code = str(listing['ZipCode'])
                    zip_codes[zip_code] = zip_codes.get(zip_code, 0) + 1
            print("Passing listings from stats")

            return {
                "total_listings": len(listings),
                "price_range": {
                    "min": min(prices) if prices else 0,
                    "max": max(prices) if prices else 0,
                    "avg": sum(prices)/len(prices) if prices else 0,
                    "data": prices
                },
                "beds_range": {
                    "min": min(beds) if beds else 0,
                    "max": max(beds) if beds else 0,
                    "avg": sum(beds)/len(beds) if beds else 0,
                    "data": beds
                },
                "baths_range": {
                    "min": min(baths) if baths else 0,
                    "max": max(baths) if baths else 0,
                    "avg": sum(baths)/len(baths) if baths else 0,
                    "data": baths
                },
                "sqft_range": {
                    "min": min(sqft) if sqft else 0,
                    "max": max(sqft) if sqft else 0,
                    "avg": sum(sqft)/len(sqft) if sqft else 0,
                    "data": sqft
                },
                "price_per_sqft": {
                    "min": min(price_per_sqft) if price_per_sqft else 0,
                    "max": max(price_per_sqft) if price_per_sqft else 0,
                    "avg": sum(price_per_sqft)/len(price_per_sqft) if price_per_sqft else 0,
                    "data": price_per_sqft
                },
                "cities": cities,
                "zip_codes": zip_codes
            }

        except Exception as e:
            print(f"Error calculating stats: {e}")
            return {
                "total_listings": len(listings),
                "price_range": {"min": 0, "max": 0, "avg": 0, "data": []},
                "beds_range": {"min": 0, "max": 0, "avg": 0, "data": []},
                "baths_range": {"min": 0, "max": 0, "avg": 0, "data": []},
                "sqft_range": {"min": 0, "max": 0, "avg": 0, "data": []},
                "price_per_sqft": {"min": 0, "max": 0, "avg": 0, "data": []},
                "cities": {},
                "zip_codes": {}
            }

    def generate_property_visualization(self, listings: List[Dict[str, Any]]) -> Optional[Dict]:
        """Generate improved visualization using Matplotlib and add descriptive text using Gemini."""
        try:
            print("Generating property visualization...")
            # Set the style and color palette
            sns.set_style("whitegrid")
            plt.rcParams.update({'font.size': 10})
            plt.rcParams['axes.prop_cycle'] = plt.cycler(color=sns.color_palette("viridis", 10))
            
            stats = self._calculate_listing_stats(listings)
            print("Stats:", stats)
            # Create a dataframe for easier plotting
            df = pd.DataFrame(listings)
            print("DF:", df)
            for col in ['Price', 'Beds', 'Baths', 'Sqft']:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: self._extract_first_number(x))
            
            # Calculate price per sqft
            if 'Price' in df.columns and 'Sqft' in df.columns:
                df['Price_Per_Sqft'] = df.apply(lambda row: row['Price'] / row['Sqft'] if row['Sqft'] > 0 else 0, axis=1)
                df = df[df['Price_Per_Sqft'] > 0]  # Filter out invalid values
            
            # Generate timestamp for both images
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_urls = {}
            print("Starting first image")
            # =========== FIRST IMAGE: PRICING ANALYTICS ===========
            fig1 = plt.figure(figsize=(16, 12))
            gs1 = GridSpec(2, 3, figure=fig1)
            
            # 1. Price Distribution - Better histogram with KDE
            ax1 = fig1.add_subplot(gs1[0, 0:2])
            if 'Price' in df.columns and not df['Price'].empty:
                sns.histplot(df['Price'], bins=15, kde=True, color='#1E88E5', ax=ax1)
                ax1.set_title('Property Price Distribution', fontweight='bold')
                ax1.set_xlabel('Price ($)')
                ax1.set_ylabel('Frequency')
                # Format x-axis with commas for thousands
                ax1.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x:,.0f}"))
                
                # Add mean line
                if not df['Price'].empty:
                    mean_price = df['Price'].mean()
                    ax1.axvline(mean_price, color='#D81B60', linestyle='--', linewidth=2, 
                               label=f'Mean: ${mean_price:,.0f}')
                    ax1.legend()
            
            # 2. Price vs. Square Footage Scatter Plot with Regression Line
            ax2 = fig1.add_subplot(gs1[0, 2])
            if 'Price' in df.columns and 'Sqft' in df.columns and not df['Price'].empty and not df['Sqft'].empty:
                sns.regplot(x='Sqft', y='Price', data=df, scatter_kws={'alpha':0.6, 's':50, 'color':'#1E88E5'}, 
                           line_kws={'color':'#D81B60'}, ax=ax2)
                ax2.set_title('Price vs. Square Footage', fontweight='bold')
                ax2.set_xlabel('Square Footage')
                ax2.set_ylabel('Price ($)')
                ax2.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x:,.0f}"))
            
            # 3. Price per Square Foot Distribution
            ax3 = fig1.add_subplot(gs1[1, 0])
            if 'Price_Per_Sqft' in df.columns and not df['Price_Per_Sqft'].empty:
                sns.boxplot(y=df['Price_Per_Sqft'], color='#00ACC1', ax=ax3)
                ax3.set_title('Price per Square Foot', fontweight='bold')
                ax3.set_ylabel('Price per Sq. Ft. ($)')
                ax3.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x:,.0f}"))
                # Adding individual data points
                sns.stripplot(y=df['Price_Per_Sqft'], color='#1A237E', alpha=0.5, size=4, ax=ax3)
            
            # 4. Price Comparison by Location (ZIP code)
            ax4 = fig1.add_subplot(gs1[1, 1:3])
            if 'ZipCode' in df.columns and 'Price' in df.columns and not df['ZipCode'].empty and not df['Price'].empty:
                # Group by ZIP code and get mean price
                zip_price = df.groupby('ZipCode')['Price'].agg(['mean', 'count']).reset_index()
                zip_price = zip_price.sort_values('mean', ascending=False)
                
                # Create bar plot
                bars = ax4.bar(zip_price['ZipCode'].astype(str), zip_price['mean'], 
                        color=sns.color_palette("viridis", len(zip_price)))
                
                # Add count as text on bars
                for i, (_, row) in enumerate(zip_price.iterrows()):
                    ax4.text(i, row['mean']/2, f"n={row['count']}", 
                            ha='center', va='center', color='white', fontweight='bold')
                
                ax4.set_title('Average Price by ZIP Code', fontweight='bold')
                ax4.set_xlabel('ZIP Code')
                ax4.set_ylabel('Average Price ($)')
                ax4.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x:,.0f}"))
                
                # Add value labels on top of bars
                ax4.bar_label(bars, fmt='${:,.0f}', padding=3)
            
            # Add super title for first image
            fig1.suptitle(f'Real Estate Pricing Analysis: {len(listings)} Properties', 
                         fontsize=20, fontweight='bold', y=0.98)
            
            # Add timestamp and footer
            fig1.text(0.5, 0.01, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} • Data Source: Property Listings API", 
                     ha='center', fontsize=9, style='italic', color='#666666')
            
            # Adjust layout and save first image
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save pricing analytics to bytes buffer
            buf1 = io.BytesIO()
            fig1.savefig(buf1, format='png', dpi=300, bbox_inches='tight')
            buf1.seek(0)
            plt.close(fig1)
            print("Pricing analytics saved to bytes buffer")
            # Upload to S3
            filename1 = f"property_pricing_analysis_{timestamp}.png"
            prefix1 = f"visualizations/property_listings/{timestamp}"
            
            pricing_url = upload_visualization_to_s3(
                buf1.getvalue(),
                prefix1,
                filename1
            )
            image_urls["pricing_analysis"] = pricing_url
            print("Pricing analytics uploaded to S3")
            # =========== SECOND IMAGE: PROPERTY CHARACTERISTICS ===========
            fig2 = plt.figure(figsize=(16, 10))
            gs2 = GridSpec(2, 3, figure=fig2)
            
            # 1. Bedrooms Count Distribution
            ax1 = fig2.add_subplot(gs2[0, 0])
            if 'Beds' in df.columns and not df['Beds'].empty:
                bed_counts = df['Beds'].value_counts().sort_index()
                bars = ax1.bar(bed_counts.index, bed_counts.values, 
                       color=sns.color_palette("viridis", len(bed_counts)))
                ax1.set_title('Bedroom Distribution', fontweight='bold')
                ax1.set_xlabel('Number of Bedrooms')
                ax1.set_ylabel('Count')
                ax1.bar_label(bars, fmt='%d')
            
            # 2. Bathrooms Count Distribution
            ax2 = fig2.add_subplot(gs2[0, 1])
            if 'Baths' in df.columns and not df['Baths'].empty:
                bath_counts = df['Baths'].value_counts().sort_index()
                bars = ax2.bar(bath_counts.index, bath_counts.values,
                       color=sns.color_palette("viridis", len(bath_counts)))
                ax2.set_title('Bathroom Distribution', fontweight='bold')
                ax2.set_xlabel('Number of Bathrooms')
                ax2.set_ylabel('Count')
                ax2.bar_label(bars, fmt='%d')
            
            # 3. Square Footage Distribution
            ax3 = fig2.add_subplot(gs2[0, 2])
            if 'Sqft' in df.columns and not df['Sqft'].empty:
                sns.histplot(df['Sqft'], bins=12, kde=True, color='#00ACC1', ax=ax3)
                ax3.set_title('Square Footage Distribution', fontweight='bold')
                ax3.set_xlabel('Square Feet')
                ax3.set_ylabel('Frequency')
                ax3.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{x:,.0f}"))
            
            # 4. Beds vs Baths Heatmap
            ax4 = fig2.add_subplot(gs2[1, 0:2])
            if 'Beds' in df.columns and 'Baths' in df.columns and not df['Beds'].empty and not df['Baths'].empty:
                # Create a cross-tabulation of beds and baths
                beds_baths = pd.crosstab(df['Beds'], df['Baths'])
                sns.heatmap(beds_baths, annot=True, cmap='viridis', fmt='d', linewidths=.5, ax=ax4)
                ax4.set_title('Bedroom vs Bathroom Configuration', fontweight='bold')
                ax4.set_xlabel('Number of Bathrooms')
                ax4.set_ylabel('Number of Bedrooms')
            
            # 5. Market Overview Stats
            ax5 = fig2.add_subplot(gs2[1, 2])
            ax5.axis('off')
            
            props = dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.8)
            overview_text = f"""
            MARKET OVERVIEW
            
            Total Properties: {stats['total_listings']}
            
            PRICE METRICS
            Minimum: ${stats['price_range']['min']:,.0f}
            Maximum: ${stats['price_range']['max']:,.0f}
            Average: ${stats['price_range']['avg']:,.0f}
            
            PROPERTY CHARACTERISTICS
            Beds: {stats['beds_range']['min']:.0f} - {stats['beds_range']['max']:.0f}
            Baths: {stats['baths_range']['min']:.0f} - {stats['baths_range']['max']:.0f}
            Sqft: {stats['sqft_range']['min']:,.0f} - {stats['sqft_range']['max']:,.0f}
            
            INVESTMENT METRICS
            Avg Price/Sqft: ${stats['price_per_sqft']['avg']:,.2f}
            """
            
            ax5.text(0.05, 0.95, overview_text, transform=ax5.transAxes, fontsize=11,
                    verticalalignment='top', bbox=props, family='monospace')
            
            # Add super title for second image
            fig2.suptitle(f'Real Estate Property Characteristics: {len(listings)} Properties', 
                         fontsize=20, fontweight='bold', y=0.98)
            
            # Add timestamp and footer
            fig2.text(0.5, 0.01, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} • Data Source: Property Listings API", 
                     ha='center', fontsize=9, style='italic', color='#666666')
            
            # Adjust layout and save second image
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Save property characteristics to bytes buffer
            buf2 = io.BytesIO()
            fig2.savefig(buf2, format='png', dpi=300, bbox_inches='tight')
            buf2.seek(0)
            plt.close(fig2)
            
            # Upload to S3
            filename2 = f"property_characteristics_{timestamp}.png"
            prefix2 = f"visualizations/property_listings/{timestamp}"
            
            characteristics_url = upload_visualization_to_s3(
                buf2.getvalue(),
                prefix2,
                filename2
            )
            image_urls["property_characteristics"] = characteristics_url
            
            # Generate descriptive text using Gemini
            viz_description = self.model.generate_content(f"""
            Describe these visualizations of {stats['total_listings']} real estate properties split into two images:
            
            IMAGE 1: PRICING ANALYTICS
            - Property price distribution histogram with density curve
            - Price vs. square footage scatter plot with regression line
            - Price per square foot box plot with data points
            - Average price by ZIP code bar chart with listing counts
            
            IMAGE 2: PROPERTY CHARACTERISTICS
            - Bedroom distribution bar chart
            - Bathroom distribution bar chart
            - Square footage distribution histogram with density curve
            - Bedroom vs bathroom configuration heatmap
            - Market overview statistics panel
            
            Market Overview:
            - {stats['total_listings']} total properties analyzed across ZIP codes: {', '.join(str(z) for z in stats['zip_codes'].keys())}
            - Price range: ${stats['price_range']['min']:,.0f} to ${stats['price_range']['max']:,.0f}, average: ${stats['price_range']['avg']:,.0f}
            - Average price per square foot: ${stats['price_per_sqft']['avg']:,.2f}
            - Bedroom range: {stats['beds_range']['min']:.0f} to {stats['beds_range']['max']:.0f}
            - Bathroom range: {stats['baths_range']['min']:.0f} to {stats['baths_range']['max']:.0f}
            - Square footage: {stats['sqft_range']['min']:,.0f} to {stats['sqft_range']['max']:,.0f} sq ft
            
            Provide a concise, professional description of what these visualizations reveal about this real estate market,
            focusing on investment insights, pricing trends, and any notable patterns. Include 2-3 specific observations.
            Limit to 500 words.
            """)
            # Return both URLs and description
            return {
                "urls": image_urls,
                "description": viz_description.text if viz_description else "No description available"
            }

        except Exception as e:
            print(f"Error generating visualization: {e}")
            return None

    def generate_detailed_summary(self, listings: List[Dict[str, Any]]) -> str:
        """Generate detailed summary using Gemini."""
        try:
            stats = self._calculate_listing_stats(listings)
            
            # Prepare listings data for the prompt by converting dates to strings
            sanitized_listings = []
            for listing in listings:
                sanitized_listing = {}
                for key, value in listing.items():
                    if isinstance(value, datetime):
                        sanitized_listing[key] = value.isoformat()
                    else:
                        sanitized_listing[key] = value
                sanitized_listings.append(sanitized_listing)

            prompt = f"""
            Analyze these real estate listings and provide a detailed market summary.
            
            Overview:
            - Total Properties: {stats['total_listings']}
            - Price Range: ${stats['price_range']['min']:,.0f} to ${stats['price_range']['max']:,.0f}
            - Average Price: ${stats['price_range']['avg']:,.0f}
            
            Property Characteristics:
            - Bedrooms: {stats['beds_range']['min']:.0f} to {stats['beds_range']['max']:.0f}
            - Bathrooms: {stats['baths_range']['min']:.0f} to {stats['baths_range']['max']:.0f}
            - Square Footage: {stats['sqft_range']['min']:,.0f} to {stats['sqft_range']['max']:,.0f}

            Detailed Listings:
            {json.dumps(sanitized_listings, indent=2, default=self._serialize_date)}

            Please provide a comprehensive analysis including:
            1. Market Overview
            2. Property Size Analysis
            3. Price Analysis
            4. Location Insights
            5. Investment Potential
            6. Notable Properties
            7. Buyer Recommendations

            Format in markdown with clear sections.
            """

            response = self.model.generate_content(prompt)
            
            if response and hasattr(response, 'text'):
                return response.text
            
            return "Unable to generate summary."

        except Exception as e:
            print(f"Error generating summary: {e}")
            return f"Error generating summary: {str(e)}" 