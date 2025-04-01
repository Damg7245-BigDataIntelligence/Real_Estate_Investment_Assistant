import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import boto3
from dotenv import load_dotenv
import time
import random
import csv
import os
from datetime import datetime
from io import StringIO

# Load environment variables from .env file
load_dotenv()

def setup_s3_client():
    """Set up and return an S3 client using credentials from .env file"""
    try:
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_region = os.getenv('AWS_REGION')  # Default to us-east-1 if not specified
        
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region
        )
        
        return s3_client
    except Exception as e:
        print(f"Error setting up S3 client: {e}")
        return None

def save_csv_locally(listings, filename):
    """Save listings to a CSV file locally."""
    try:
        fieldnames = ["city", "address", "StateName", "RegionName", "price", "beds", "baths", "sqft", "url", "date_scraped"]
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for listing in listings:
                writer.writerow(listing)
        print(f"Saved CSV file locally as {filename}")
        return True
    except Exception as e:
        print(f"Error saving CSV locally: {e}")
        return False

def upload_csv_to_s3(file_path, bucket_name, s3_key):
    """Upload the local CSV file directly to S3.
       This function uses a constant S3 key so that the file is overwritten each time.
    """
    try:
        s3_client = setup_s3_client()
        if not s3_client:
            print("Failed to set up S3 client. CSV will not be uploaded.")
            return False
        
        print(f"Uploading {file_path} to s3://{bucket_name}/{s3_key} (this will overwrite the file)...")
        s3_client.upload_file(file_path, bucket_name, s3_key)
        print("Successfully uploaded CSV file to S3.")
        return True
    except Exception as e:
        print(f"Error uploading CSV to S3: {e}")
        return False

def random_delay(min_seconds=1, max_seconds=5):
    """Add a random delay to mimic human behavior"""
    time.sleep(random.uniform(min_seconds, max_seconds))

def setup_driver():
    """Set up the undetected ChromeDriver with stealth settings"""
    options = uc.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--start-maximized")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    options.add_argument(f"user-agent={user_agent}")
    options.add_argument("--window-size=1920,1080")
    
    driver = uc.Chrome(options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver

def check_for_visible_captcha(driver):
    """Check if a CAPTCHA is actually visible on the page"""
    try:
        captcha_indicators = [
            "//div[contains(@class, 'captcha') and not(contains(@style, 'display: none'))]",
            "//iframe[contains(@src, 'captcha') or contains(@src, 'recaptcha')]",
            "//button[contains(text(), 'not a robot') or contains(text(), 'verify')]",
            "//div[contains(@class, 'challenge')]",
            "//div[contains(@id, 'captcha')]"
        ]
        
        for indicator in captcha_indicators:
            try:
                elements = driver.find_elements(By.XPATH, indicator)
                for element in elements:
                    if element.is_displayed():
                        print(f"Visible CAPTCHA element found: {indicator}")
                        return True
            except:
                continue
        
        iframes = driver.find_elements(By.TAG_NAME, "iframe")
        if iframes:
            print(f"Found {len(iframes)} iframes, checking for CAPTCHAs inside...")
            for i, iframe in enumerate(iframes):
                try:
                    driver.switch_to.frame(iframe)
                    for indicator in captcha_indicators:
                        try:
                            elements = driver.find_elements(By.XPATH, indicator)
                            for element in elements:
                                if element.is_displayed():
                                    print(f"Visible CAPTCHA found in iframe {i+1}")
                                    driver.switch_to.default_content()
                                    return True
                        except:
                            continue
                    driver.switch_to.default_content()
                except:
                    driver.switch_to.default_content()
                    continue
        
        if "captcha" in driver.page_source.lower() or "robot" in driver.page_source.lower():
            print("CAPTCHA keywords found in page source, but no visible CAPTCHA elements detected.")
            
        return False
        
    except Exception as e:
        print(f"Error checking for CAPTCHA: {e}")
        return False

def try_solve_captcha(driver):
    """Attempt to solve a CAPTCHA if present"""
    try:
        if not check_for_visible_captcha(driver):
            print("No visible CAPTCHA detected. Continuing...")
            return False
            
        print("Visible CAPTCHA detected! Attempting to solve automatically...")
        
        iframes = driver.find_elements(By.TAG_NAME, "iframe")
        captcha_frames = [None] + iframes
        
        for frame in captcha_frames:
            try:
                if frame:
                    driver.switch_to.frame(frame)
                
                captcha_selectors = [
                    ".captcha-slider-button", 
                    "[role='slider']",
                    ".captcha-verify",
                    ".recaptcha-checkbox-border",
                    "[data-testid='captcha-slider']",
                    "//button[contains(@class, 'captcha')]",
                    "//div[contains(@class, 'slider') and @role='button']"
                ]
                
                for selector in captcha_selectors:
                    try:
                        button = driver.find_element(By.XPATH, selector) if selector.startswith("//") else driver.find_element(By.CSS_SELECTOR, selector)
                        if button and button.is_displayed():
                            print(f"Found CAPTCHA button using: {selector}")
                            actions = ActionChains(driver)
                            print("Clicking and holding CAPTCHA button...")
                            actions.click_and_hold(button).perform()
                            print("Waiting for CAPTCHA verification (8 seconds)...")
                            time.sleep(8)
                            actions.release().perform()
                            time.sleep(3)
                            
                            if frame:
                                driver.switch_to.default_content()
                                
                            if not check_for_visible_captcha(driver):
                                print("CAPTCHA appears to be solved automatically!")
                                return True
                            else:
                                print("Automatic CAPTCHA solving failed.")
                                if frame:
                                    driver.switch_to.default_content()
                    except Exception as e:
                        if "no such element" in str(e).lower():
                            pass
                        else:
                            print(f"Error with CAPTCHA selector {selector}: {str(e)[:100]}...")
                
                if frame:
                    driver.switch_to.default_content()
                    
            except Exception as e:
                print(f"Error processing frame: {e}")
                driver.switch_to.default_content()
        
        print("CAPTCHA detected but could not be solved automatically.")
        input("Please solve the CAPTCHA in the browser window and press Enter when done...")
        random_delay(3, 5)
        return True
            
    except Exception as e:
        print(f"Error in CAPTCHA handling: {e}")
        try:
            driver.switch_to.default_content()
        except:
            pass
    
    return False

def search_redfin_for_city(driver, city, state="MA"):
    """Navigate to Redfin and search for properties in the specified city"""
    try:
        print(f"Searching for {city}, {state} on Redfin...")
        driver.get("https://www.redfin.com/")
        random_delay(3, 5)
        
        try:
            window_size = driver.get_window_size()
            center_x = window_size['width'] // 2
            center_y = window_size['height'] // 2
            actions = ActionChains(driver)
            actions.move_to_location(center_x, center_y).perform()
            random_delay(0.5, 1.5)
            for _ in range(3):
                x, y = random.randint(-100, 100), random.randint(-100, 100)
                actions.move_by_offset(x, y).perform()
                random_delay(0.5, 1.5)
        except Exception as e:
            print(f"Mouse movement error: {e}")
            
        print("Looking for search box...")
        search_selectors = [
            "input[placeholder='City, Address, School, Agent, ZIP']",
            "input[type='search']",
            "[data-id='search-box-input']",
            "#search-box-input"
        ]
        
        search_box = None
        for selector in search_selectors:
            try:
                search_box = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                if search_box:
                    print(f"Found search box using: {selector}")
                    break
            except:
                continue
        
        if not search_box:
            print("Could not locate search box.")
            return False
        
        search_box.click()
        random_delay(0.5, 1.5)
        print(f"Typing search query for {city}...")
        search_box.clear()
        search_term = f"{city}, {state}"
        for char in search_term:
            search_box.send_keys(char)
            random_delay(0.05, 0.15)
        
        random_delay(1, 2)
        
        try:
            city_xpath = city.replace("'", "\\'")
            suggestion = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, f"//div[contains(text(), '{city_xpath}, {state}')]"))
            )
            print(f"Found {city} suggestion, clicking it...")
            suggestion.click()
        except:
            print("No suggestions appeared, pressing Enter...")
            search_box.send_keys(Keys.RETURN)
        
        print("Waiting for results to load...")
        try:
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".HomeCardContainer"))
            )
        except Exception as e:
            print(f"Error waiting for results page: {e}")
            return False
        
        random_delay(5, 8)
        try_solve_captcha(driver)
        
        for _ in range(3):
            driver.execute_script(f"window.scrollBy(0, {random.randint(300, 800)});")
            random_delay(1, 3)
        
        print(f"Current URL after search: {driver.current_url}")
        return True
    
    except Exception as e:
        print(f"An error occurred during search for {city}: {e}")
        return False

def parse_address(address):
    """
    Parse an address string to extract state and region (zip code).
    Example: "9 M St #5, Boston, MA 02127" returns ("MA", "02127")
    """
    state = "N/A"
    region = "N/A"
    try:
        # Split the address by commas
        parts = address.split(',')
        if len(parts) >= 3:
            # The last part is expected to contain state and zip code.
            last_part = parts[-1].strip()  # e.g., "MA 02127"
            # Split by space
            tokens = last_part.split()
            if len(tokens) >= 2:
                state = tokens[0]
                region = tokens[1]
    except Exception as e:
        print(f"Error parsing address: {e}")
    return state, region

def extract_listings(driver, current_city):
    """Extract property listings from Redfin and add city information along with state and region fields"""
    listings = []
    seen_urls = set()
    
    try:
        print("Searching for property cards...")
        property_cards = driver.find_elements(By.XPATH, "//div[contains(@class, 'HomeCardContainer') or contains(@class, 'HomeCard')]")
        
        if not property_cards:
            property_cards = driver.find_elements(By.XPATH, "//div[contains(@class, 'bp-Homecard')]")
        if not property_cards:
            property_cards = driver.find_elements(By.XPATH, 
                "//div[contains(@class, 'HomeCard') or contains(@class, 'SearchResultItem') or contains(@role, 'button')]")
        if not property_cards:
            print("No property cards found. Skipping extraction for this page.")
            return []
            
        print(f"Found {len(property_cards)} property cards in {current_city}. Extracting data...")
        
        for i, card in enumerate(property_cards):
            try:
                if not card.is_displayed():
                    continue
                    
                url = "N/A"
                try:
                    link_elem = card.find_element(By.TAG_NAME, "a")
                    url = link_elem.get_attribute("href")
                except:
                    try:
                        url_elems = card.find_elements(By.CSS_SELECTOR, "a[href*='/homes/']")
                        if url_elems:
                            url = url_elems[0].get_attribute("href")
                    except:
                        pass
                
                if url == "N/A" or url in seen_urls:
                    continue
                
                if url != "N/A":
                    seen_urls.add(url)
                
                address = None
                try:
                    addr_div = card.find_element(By.XPATH, ".//div[contains(@class, 'Homecard__Address') or contains(@class, 'homeAddressV2')]")
                    if addr_div:
                        address = addr_div.text.strip()
                except:
                    pass
                if not address:
                    try:
                        addr_elem = card.find_element(By.TAG_NAME, "address")
                        if addr_elem:
                            address = addr_elem.get_attribute("textContent").strip()
                    except:
                        pass
                if not address:
                    try:
                        aria_label = card.get_attribute("aria-label")
                        if aria_label and "Property at" in aria_label:
                            address = aria_label.split("Property at")[1].split(",")[0].strip()
                    except:
                        pass
                if not address:
                    try:
                        text_elems = card.find_elements(By.XPATH, ".//*[not(self::script) and not(self::style)]")
                        for elem in text_elems:
                            text = elem.text.strip()
                            if text and any(street_suffix in text.lower() for street_suffix in 
                                          [" st", " ave", " rd", " dr", " ln", " way", " ct", " pl"]):
                                address = text
                                break
                    except:
                        pass
                if not address:
                    print(f"No address found for {current_city} card {i+1}, skipping")
                    continue
                
                # Parse address for state and region (zip code)
                state_name, region_name = parse_address(address)
                
                price = "N/A"
                try:
                    price_elem = card.find_element(By.XPATH, 
                                                 ".//*[contains(@class, 'Price--value') or contains(@class, 'homePriceV2') or contains(@class, 'Homecard__Price')]")
                    if price_elem:
                        price = price_elem.text.strip()
                except:
                    try:
                        elements = card.find_elements(By.XPATH, ".//*")
                        for elem in elements:
                            text = elem.text.strip()
                            if text and text.startswith("$") and any(c.isdigit() for c in text):
                                price = text
                                break
                    except:
                        pass
                
                beds, baths, sqft = "N/A", "N/A", "N/A"
                try:
                    beds_elem = card.find_element(By.XPATH, 
                                               ".//*[contains(@class, 'beds') or contains(@class, 'Beds') or contains(text(), 'bed')]")
                    if beds_elem:
                        beds = ''.join(c for c in beds_elem.text.strip() if c.isdigit() or c == '.')
                except:
                    pass
                try:
                    baths_elem = card.find_element(By.XPATH, 
                                                ".//*[contains(@class, 'baths') or contains(@class, 'Baths') or contains(text(), 'bath')]")
                    if baths_elem:
                        baths = ''.join(c for c in baths_elem.text.strip() if c.isdigit() or c == '.')
                except:
                    pass
                try:
                    sqft_elem = card.find_element(By.XPATH, 
                                               ".//*[contains(@class, 'sqft') or contains(@class, 'Sqft') or contains(text(), 'sq ft')]")
                    if sqft_elem:
                        sqft = ''.join(c for c in sqft_elem.text.strip() if c.isdigit() or c == ',')
                except:
                    pass
                
                listing = {
                    "city": current_city,
                    "address": address,
                    "StateName": state_name,
                    "RegionName": region_name,
                    "price": price,
                    "beds": beds,
                    "baths": baths,
                    "sqft": sqft,
                    "url": url,
                    "date_scraped": datetime.now().strftime("%Y-%m-%d")
                }
                
                listings.append(listing)
                print(f"Extracted {current_city} listing {i+1}: {address}")
                if i % 5 == 0:
                    random_delay(0.3, 0.8)
                
            except Exception as e:
                print(f"Error extracting {current_city} listing {i+1}: {e}")
                continue
        
        unique_listings = []
        seen = set()
        for listing in listings:
            if listing['url'] not in seen and listing['url'] != "N/A":
                seen.add(listing['url'])
                unique_listings.append(listing)
            
        print(f"Removed {len(listings) - len(unique_listings)} duplicate listings from {current_city}")
        return unique_listings
        
    except Exception as e:
        print(f"An error occurred during extraction for {current_city}: {e}")
    return listings

def navigate_to_next_page(driver, current_page):
    """Navigate to the next page of search results on Redfin"""
    try:
        print(f"Attempting to navigate to page {current_page + 1}...")
        next_button_selectors = [
            ".goToPage[aria-label='Next']",
            "button.nextButton",
            "a.nextButton",
            "[data-rf-test-id='react-pagination-next']",
            "button.goToPage[title='Next page']",
            "//button[contains(text(), 'Next')]",
            "//a[contains(text(), 'Next')]"
        ]
        
        for selector in next_button_selectors:
            try:
                next_button = driver.find_element(By.XPATH, selector) if selector.startswith("//") else driver.find_element(By.CSS_SELECTOR, selector)
                if next_button and next_button.is_displayed() and next_button.is_enabled():
                    print(f"Found 'Next' button using selector: {selector}")
                    driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", next_button)
                    random_delay(1, 2)
                    driver.execute_script("arguments[0].click();", next_button)
                    print(f"Clicked 'Next' button, navigating to page {current_page + 1}...")
                    random_delay(5, 8)
                    try:
                        WebDriverWait(driver, 15).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, ".HomeCardContainer"))
                        )
                        return True
                    except:
                        print("Failed to load property cards on the new page.")
                        return False
            except:
                continue
        
        print("Looking for pagination elements...")
        pagination_selectors = [
            "nav.PagingControls",
            ".Pagination",
            "[data-rf-test-id='react-pagination']"
        ]
        
        for selector in pagination_selectors:
            try:
                pagination = driver.find_element(By.CSS_SELECTOR, selector)
                if pagination:
                    next_page_num = str(current_page + 1)
                    page_links = pagination.find_elements(By.CSS_SELECTOR, "a.goToPage, button.goToPage")
                    for link in page_links:
                        if link.text.strip() == next_page_num:
                            print(f"Found link to page {next_page_num}")
                            driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", link)
                            random_delay(1, 2)
                            driver.execute_script("arguments[0].click();", link)
                            print(f"Clicked link to page {next_page_num}...")
                            random_delay(5, 8)
                            try:
                                WebDriverWait(driver, 15).until(
                                    EC.presence_of_element_located((By.CSS_SELECTOR, ".HomeCardContainer"))
                                )
                                return True
                            except:
                                print("Failed to load property cards on the new page.")
                                return False
            except:
                continue
                
        print(f"Could not find a way to navigate to page {current_page + 1}")
        return False
        
    except Exception as e:
        print(f"Error navigating to next page: {e}")
        return False

def main():
    """Main function to run the scraper for multiple cities, store CSV locally and upload to S3 (overwriting the file)"""
    ma_cities = [
        "Boston", "Dorchester", "Revere", "Chelsea"
    ]
    
    s3_bucket = os.getenv('AWS_S3_BUCKET_NAME')
    if not s3_bucket:
        print("Warning: AWS_S3_BUCKET_NAME not found in .env file. S3 upload will be skipped.")
    
    city_results = {}
    all_listings = []
    driver = setup_driver()
    
    try:
        for city in ma_cities:
            print(f"\n\n{'=' * 60}")
            print(f"STARTING SCRAPE FOR {city.upper()}, MA")
            print(f"{'=' * 60}\n")
            city_seen_urls = set()
            max_pages = 5
            search_successful = search_redfin_for_city(driver, city)
            
            if search_successful:
                for current_page in range(1, max_pages + 1):
                    print(f"\n--- Processing {city}: Page {current_page} of {max_pages} ---\n")
                    page_listings = extract_listings(driver, city)
                    
                    if page_listings:
                        print(f"Extracted {len(page_listings)} listings from {city}, page {current_page}.")
                        unique_listings = []
                        for listing in page_listings:
                            if listing['url'] != 'N/A' and listing['url'] not in city_seen_urls:
                                city_seen_urls.add(listing['url'])
                                unique_listings.append(listing)
                        all_listings.extend(unique_listings)
                        print(f"Added {len(unique_listings)} unique listings from this page")
                        print(f"Total listings across all cities: {len(all_listings)}")
                    else:
                        print(f"No listings found for {city} on page {current_page}.")
                    
                    if current_page == max_pages:
                        print(f"Reached maximum number of pages ({max_pages}) for {city}. Moving to next city.")
                        break
                    
                    next_page_success = navigate_to_next_page(driver, current_page)
                    if not next_page_success:
                        print(f"Could not navigate to {city} page {current_page + 1}. Moving to next city.")
                        break
                    
                    try_solve_captcha(driver)
                    random_delay(2, 4)
                
                city_results[city] = len([listing for listing in all_listings if listing['city'] == city])
            else:
                print(f"Failed to search for {city} properties on Redfin.")
                city_results[city] = 0
                
            print(f"\nCompleted scraping for {city}. Waiting before moving to next city...")
            random_delay(5, 10)
        
        if all_listings:
            # Generate a timestamp-based filename for local storage
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            local_filename = f"massachusetts_redfin_listings_{timestamp}.csv"
            
            # Save the CSV file locally
            local_save_success = save_csv_locally(all_listings, local_filename)
            
            # Use a constant S3 key so that the file is overwritten in S3 (e.g., without timestamp)
            s3_key = "current_listings/massachusetts_redfin_listings.csv"
            
            if s3_bucket:
                s3_upload_success = upload_csv_to_s3(local_filename, s3_bucket, s3_key)
                if s3_upload_success:
                    print(f"CSV file successfully uploaded to s3://{s3_bucket}/{s3_key}")
                else:
                    print("Failed to upload CSV file to S3.")
            else:
                print("No S3 bucket specified; skipping CSV upload.")
        else:
            print("No listings were extracted from any city.")
        
        print("\n\n" + "=" * 60)
        print("SCRAPING SUMMARY REPORT")
        print("=" * 60)
        print(f"Total cities processed: {len(ma_cities)}")
        total_listings = sum(city_results.values())
        print(f"Total listings collected: {total_listings}")
        print("\nListings per city:")
        for city, count in city_results.items():
            print(f"  {city}: {count} listings")
        
    except Exception as e:
        print(f"An error occurred in the main process: {e}")
    
    finally:
        # Explicitly quit and delete the driver to avoid __del__ warnings
        try:
            driver.quit()
        except Exception as e:
            print(f"Error during driver.quit(): {e}")
        finally:
            del driver

if __name__ == "__main__":
    main()
