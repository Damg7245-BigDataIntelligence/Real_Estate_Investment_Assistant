import google.generativeai as genai
from dotenv import load_dotenv
import os
import base64
import io
from PIL import Image
import json
import re
from typing import Dict, List, Any, Optional
from s3_utils import upload_visualization_to_s3
from datetime import datetime

load_dotenv()

class GeminiVisualizationAgent:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
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
            prices = []
            beds = []
            baths = []
            sqft = []

            for listing in listings:
                # Handle Price
                if 'Price' in listing and listing['Price']:
                    price_val = self._extract_first_number(listing['Price'])
                    if price_val > 0:
                        prices.append(price_val)

                # Handle Beds
                if 'Beds' in listing and listing['Beds']:
                    bed_val = self._extract_first_number(str(listing['Beds']))
                    if bed_val > 0:
                        beds.append(bed_val)

                # Handle Baths
                if 'Baths' in listing and listing['Baths']:
                    bath_val = self._extract_first_number(str(listing['Baths']))
                    if bath_val > 0:
                        baths.append(bath_val)

                # Handle Sqft
                if 'Sqft' in listing and listing['Sqft']:
                    sqft_val = self._extract_first_number(str(listing['Sqft']))
                    if sqft_val > 0:
                        sqft.append(sqft_val)

            return {
                "total_listings": len(listings),
                "price_range": {
                    "min": min(prices) if prices else 0,
                    "max": max(prices) if prices else 0,
                    "avg": sum(prices)/len(prices) if prices else 0
                },
                "beds_range": {
                    "min": min(beds) if beds else 0,
                    "max": max(beds) if beds else 0,
                    "avg": sum(beds)/len(beds) if beds else 0
                },
                "baths_range": {
                    "min": min(baths) if baths else 0,
                    "max": max(baths) if baths else 0,
                    "avg": sum(baths)/len(baths) if baths else 0
                },
                "sqft_range": {
                    "min": min(sqft) if sqft else 0,
                    "max": max(sqft) if sqft else 0,
                    "avg": sum(sqft)/len(sqft) if sqft else 0
                }
            }

        except Exception as e:
            print(f"Error calculating stats: {e}")
            return {
                "total_listings": len(listings),
                "price_range": {"min": 0, "max": 0, "avg": 0},
                "beds_range": {"min": 0, "max": 0, "avg": 0},
                "baths_range": {"min": 0, "max": 0, "avg": 0},
                "sqft_range": {"min": 0, "max": 0, "avg": 0}
            }

    def generate_property_visualization(self, listings: List[Dict[str, Any]]) -> Optional[str]:
        """Generate visualization using Matplotlib and add descriptive text using Gemini."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            stats = self._calculate_listing_stats(listings)
            
            # Create a figure with multiple subplots
            fig = plt.figure(figsize=(15, 10))
            fig.suptitle('Real Estate Listings Analysis', fontsize=16)
            
            # 1. Price Distribution
            plt.subplot(2, 2, 1)
            prices = [self._extract_first_number(listing['Price']) for listing in listings if 'Price' in listing]
            sns.histplot(prices, bins=10)
            plt.title('Price Distribution')
            plt.xlabel('Price ($)')
            plt.ylabel('Count')
            
            # 2. Beds vs Baths Scatter
            plt.subplot(2, 2, 2)
            beds = [self._extract_first_number(listing['Beds']) for listing in listings if 'Beds' in listing]
            baths = [self._extract_first_number(listing['Baths']) for listing in listings if 'Baths' in listing]
            plt.scatter(beds, baths, alpha=0.5)
            plt.title('Beds vs Baths')
            plt.xlabel('Bedrooms')
            plt.ylabel('Bathrooms')
            
            # 3. Square Footage Distribution
            plt.subplot(2, 2, 3)
            sqft = [self._extract_first_number(listing['Sqft']) for listing in listings if 'Sqft' in listing]
            sns.histplot(sqft, bins=10)
            plt.title('Square Footage Distribution')
            plt.xlabel('Square Feet')
            plt.ylabel('Count')
            
            # 4. Stats Summary
            plt.subplot(2, 2, 4)
            plt.axis('off')
            summary_text = f"""
            Listings Summary:
            
            Total Properties: {stats['total_listings']}
            
            Price Range:
            Min: ${stats['price_range']['min']:,.0f}
            Max: ${stats['price_range']['max']:,.0f}
            Avg: ${stats['price_range']['avg']:,.0f}
            
            Property Sizes:
            Beds: {stats['beds_range']['min']:.0f} - {stats['beds_range']['max']:.0f}
            Baths: {stats['baths_range']['min']:.0f} - {stats['baths_range']['max']:.0f}
            Sqft: {stats['sqft_range']['min']:,.0f} - {stats['sqft_range']['max']:,.0f}
            """
            plt.text(0.1, 0.1, summary_text, fontsize=10, verticalalignment='top')
            
            # Adjust layout and save
            plt.tight_layout()
            
            # Save to bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            
            # Upload to S3
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"property_visualization_{timestamp}.png"
            prefix = f"visualizations/property_listings/{timestamp}"
            
            presigned_url = upload_visualization_to_s3(
                buf.getvalue(),
                prefix,
                filename
            )
            
            # Generate descriptive text using Gemini
            viz_description = self.model.generate_content(f"""
            Describe this visualization of {stats['total_listings']} real estate listings:
            
            - Price range from ${stats['price_range']['min']:,.0f} to ${stats['price_range']['max']:,.0f}
            - Average price: ${stats['price_range']['avg']:,.0f}
            - Bedroom range: {stats['beds_range']['min']:.0f} to {stats['beds_range']['max']:.0f}
            - Bathroom range: {stats['baths_range']['min']:.0f} to {stats['baths_range']['max']:.0f}
            - Square footage: {stats['sqft_range']['min']:,.0f} to {stats['sqft_range']['max']:,.0f} sq ft
            
            Provide a brief, professional description of what the visualization shows.
            """)
            
            # Return both URL and description
            return {
                "url": presigned_url,
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