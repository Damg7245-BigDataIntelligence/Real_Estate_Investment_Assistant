import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
import time
import random
import csv
import os
import json
import urllib.parse

def random_delay(min_seconds=1, max_seconds=5):
    """Add a random delay to mimic human behavior"""
    time.sleep(random.uniform(min_seconds, max_seconds))

def setup_driver():
    """Set up the undetected ChromeDriver with stealth settings"""
    options = uc.ChromeOptions()
    
    # Only use compatible arguments
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--start-maximized")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    # Use a realistic user agent
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    options.add_argument(f"user-agent={user_agent}")
    
    # Make browser window size realistic
    options.add_argument("--window-size=1920,1080")
    
    # Create and return the driver
    driver = uc.Chrome(options=options)
    
    # Apply additional stealth techniques via JavaScript
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    return driver

def check_for_visible_captcha(driver):
    """Check if a CAPTCHA is actually visible on the page"""
    try:
        # Take a screenshot for manual verification
        driver.save_screenshot("page_check.png")
        
        # Check for visible CAPTCHA elements with common patterns
        captcha_indicators = [
            # Visual elements that would be visible
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
        
        # Check for CAPTCHA in iframes
        iframes = driver.find_elements(By.TAG_NAME, "iframe")
        if iframes:
            print(f"Found {len(iframes)} iframes, checking for CAPTCHAs inside...")
            
            for i, iframe in enumerate(iframes):
                try:
                    # Switch to iframe
                    driver.switch_to.frame(iframe)
                    
                    # Look for CAPTCHA elements in this iframe
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
                    
                    # Switch back to main content
                    driver.switch_to.default_content()
                except:
                    driver.switch_to.default_content()
                    continue
        
        # If we got here, we likely have a false positive
        # Check if keywords exist but no visual elements
        if "captcha" in driver.page_source.lower() or "robot" in driver.page_source.lower():
            print("CAPTCHA keywords found in page source, but no visible CAPTCHA elements detected.")
            print("This may be a false positive or a hidden CAPTCHA mechanism.")
            
        return False
        
    except Exception as e:
        print(f"Error checking for CAPTCHA: {e}")
        return False

def try_solve_captcha(driver):
    """Attempt to solve a CAPTCHA if present"""
    try:
        # First, check if a CAPTCHA is actually visible
        if not check_for_visible_captcha(driver):
            print("No visible CAPTCHA detected. Continuing...")
            return False
            
        print("Visible CAPTCHA detected! Attempting to solve automatically...")
        
        # Search in all potential iframes
        iframes = driver.find_elements(By.TAG_NAME, "iframe")
        
        # Try to solve in main frame first, then check iframes
        captcha_frames = [None] + iframes  # None represents the main frame
        
        for frame in captcha_frames:
            try:
                if frame:
                    driver.switch_to.frame(frame)
                
                # Try to find the slider or button element with various selectors
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
                        # Try CSS selector first
                        button = None
                        if selector.startswith("//"):
                            button = driver.find_element(By.XPATH, selector)
                        else:
                            button = driver.find_element(By.CSS_SELECTOR, selector)
                            
                        if button and button.is_displayed():
                            print(f"Found CAPTCHA button using: {selector}")
                            
                            # For hold-type CAPTCHAs
                            actions = ActionChains(driver)
                            
                            # Click and hold the button
                            print("Clicking and holding CAPTCHA button...")
                            actions.click_and_hold(button)
                            actions.perform()
                            
                            # Wait for progress bar to complete (8 seconds for Zillow)
                            print("Waiting for CAPTCHA verification (8 seconds)...")
                            time.sleep(8)
                            
                            # Release the button
                            actions.release()
                            actions.perform()
                            
                            # Wait to see if CAPTCHA is solved
                            time.sleep(3)
                            
                            # Return to main content if we were in an iframe
                            if frame:
                                driver.switch_to.default_content()
                                
                            # Check if CAPTCHA is still present
                            if not check_for_visible_captcha(driver):
                                print("CAPTCHA appears to be solved automatically!")
                                return True
                            else:
                                print("Automatic CAPTCHA solving failed.")
                                if frame:
                                    driver.switch_to.default_content()
                    except Exception as e:
                        if "no such element" in str(e).lower():
                            # No element found with this selector, try the next one
                            pass
                        else:
                            print(f"Error with CAPTCHA selector {selector}: {str(e)[:100]}...")
                
                # Return to main content if we were in an iframe
                if frame:
                    driver.switch_to.default_content()
                    
            except Exception as e:
                print(f"Error processing frame: {e}")
                # Make sure we get back to the main content
                driver.switch_to.default_content()
        
        # If automatic solving failed or no CAPTCHA elements found, ask for manual help
        print("CAPTCHA detected but could not be solved automatically.")
        input("Please solve the CAPTCHA in the browser window and press Enter when done...")
        random_delay(3, 5)
        return True
            
    except Exception as e:
        print(f"Error in CAPTCHA handling: {e}")
        # Make sure we're back to the main content
        try:
            driver.switch_to.default_content()
        except:
            pass
    
    return False

def search_zillow_for_ma(driver):
    """Navigate to Zillow and search for Massachusetts properties"""
    try:
        # Try both approaches - direct URL
        print("Navigating directly to Massachusetts listings...")
        driver.get("https://www.zillow.com/ma/")
        
        # Random delay
        random_delay(3, 5)
        
        # Add human-like behavior - move mouse and scroll
        actions = ActionChains(driver)
        for _ in range(3):
            x, y = random.randint(100, 700), random.randint(100, 500)
            actions.move_by_offset(x, y).perform()
            random_delay(0.5, 1.5)
        
        driver.execute_script(f"window.scrollBy(0, {random.randint(200, 500)});")
        random_delay(1, 3)
        
        # Check if we landed on a Massachusetts page or need to try search
        if "Enter an address" in driver.page_source and "ma" not in driver.current_url.lower():
            print("Direct URL didn't work, trying search approach...")
            
            # Navigate to Zillow homepage
            driver.get("https://www.zillow.com/")
            random_delay(3, 5)
            
            # Find and click on the search box
            print("Looking for search box...")
            search_selectors = [
                "input[placeholder='Enter an address, neighborhood, city, or ZIP code']",
                "input[role='combobox']",
                ".Input-c11n-8-106-0__sc-4ry0fw-0"
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
                driver.save_screenshot("zillow_error.png")
                return False
            
            # Click and type in search box
            search_box.click()
            random_delay(0.5, 1.5)
            
            print("Typing search query...")
            search_box.clear()
            search_term = "Massachusetts homes for sale"
            for char in search_term:
                search_box.send_keys(char)
                random_delay(0.05, 0.15)  # Faster typing
            
            random_delay(1, 2)
            
            # Try to find and click the submit button
            try:
                submit_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit']"))
                )
                submit_button.click()
            except:
                # Fall back to pressing Enter key
                search_box.send_keys(Keys.RETURN)
        
        # Wait for results to load
        print("Waiting for results to load...")
        try:
            # Wait for property cards to appear
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "ul.photo-cards"))
            )
        except Exception as e:
            print(f"Error waiting for results page: {e}")
            print(f"Current URL: {driver.current_url}")
            driver.save_screenshot("results_error.png")
            
            # Try to find any listing element as a fallback
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "[data-test='property-card']"))
                )
                print("Found property cards using alternative selector.")
            except:
                return False
        
        # Wait for the page to fully load
        print("Allowing extra time for page to fully load...")
        random_delay(5, 8)
        
        # Check for CAPTCHA
        try_solve_captcha(driver)
        
        # Scroll to load more content
        for _ in range(3):
            driver.execute_script(f"window.scrollBy(0, {random.randint(300, 800)});")
            random_delay(1, 3)
        
        # Save the current URL for reference
        print(f"Current URL after search: {driver.current_url}")
        
        # Take screenshot of the results page
        driver.save_screenshot("search_results.png")
            
        return True
    
    except Exception as e:
        print(f"An error occurred during search: {e}")
        driver.save_screenshot("search_error.png")
        return False

def extract_listings(driver):
    """Extract property listings using the main container element"""
    listings = []
    
    try:
        # Target the main container directly
        print("Looking for the main property list container...")
        
        # First try to find the exact container we identified
        try:
            main_container = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "ul.photo-cards"))
            )
            print("Found main property list container.")
        except:
            print("Could not find the main container. Trying alternative selectors...")
            
            # Try alternative selectors if the main one fails
            container_selectors = [
                "#grid-search-results", 
                "ul.List-c11n-8-109-3__sc-1smrmqp-0",
                "[data-testid='search-results']",
                "div[id^='search-results']"
            ]
            
            main_container = None
            for selector in container_selectors:
                try:
                    main_container = driver.find_element(By.CSS_SELECTOR, selector)
                    if main_container:
                        print(f"Found container using alternative selector: {selector}")
                        break
                except:
                    continue
        
        if not main_container:
            print("Could not find any listing container.")
            driver.save_screenshot("no_container.png")
            with open("page_source.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            return []
        
        # Find all property cards - try multiple selector patterns
        card_selectors = [
            "li", 
            "li.StyledListCardWrapper-srp-8-109-3__sc-wtsrtn-0",
            "li[data-test='property-card']",
            "div[data-test='property-card']",
            "article[role='group']"
        ]
        
        property_cards = []
        for selector in card_selectors:
            try:
                cards = main_container.find_elements(By.CSS_SELECTOR, selector)
                if cards and len(cards) > 0:
                    property_cards = cards
                    print(f"Found {len(cards)} property cards using selector: {selector}")
                    break
            except:
                continue
                
        if not property_cards:
            print("Could not find any property cards in the container.")
            return []
        
        print(f"Extracting data from {len(property_cards)} property cards...")
        
        # Process each property card
        for i, card in enumerate(property_cards):
            try:
                # Skip advertisement cards
                if card.get_attribute("class") and "nav-ad-empty" in card.get_attribute("class"):
                    continue
                
                # Scroll to the card for better visibility
                driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", card)
                random_delay(0.2, 0.5)
                
                # Extract address using multiple selectors
                address = "N/A"
                address_selectors = [
                    "[data-test='property-card-addr']",
                    "address",
                    ".StyledPropertyCardDataArea-srp__sc-uhp67s-0 address",
                    "a[data-test='property-card-link']"
                ]
                
                for selector in address_selectors:
                    try:
                        addr_elem = card.find_element(By.CSS_SELECTOR, selector)
                        if addr_elem and addr_elem.text.strip():
                            address = addr_elem.text.strip()
                            break
                    except:
                        continue
                
                # Skip if no address found
                if address == "N/A":
                    print(f"No address found for card {i+1}, skipping")
                    continue
                
                # Extract price
                price = "N/A"
                price_selectors = [
                    "[data-test='property-card-price']",
                    ".StyledPropertyCardDataArea-srp__sc-uhp67s-0 span:first-child",
                    "span.PropertyCardPrice"
                ]
                
                for selector in price_selectors:
                    try:
                        price_elem = card.find_element(By.CSS_SELECTOR, selector)
                        if price_elem and price_elem.text.strip():
                            price = price_elem.text.strip()
                            break
                    except:
                        continue
                
                # Extract property details (beds, baths, sqft)
                beds, baths, sqft = "N/A", "N/A", "N/A"
                
                # First try to find the details container
                detail_container_selectors = [
                    "ul[data-test='property-card-details']",
                    ".StyledPropertyCardHomeDetails-srp__sc-1ci9ij9-0",
                    "ul.StyledPropertyCardHomeDetailsList-c11n-8-109-3__sc-1j0som5-0"
                ]
                
                for selector in detail_container_selectors:
                    try:
                        details_container = card.find_element(By.CSS_SELECTOR, selector)
                        if details_container:
                            details = details_container.find_elements(By.TAG_NAME, "li")
                            if len(details) >= 1:
                                beds = details[0].text.strip()
                            if len(details) >= 2:
                                baths = details[1].text.strip()
                            if len(details) >= 3:
                                sqft = details[2].text.strip()
                            break
                    except:
                        continue
                
                # Extract URL
                url = "N/A"
                link_selectors = [
                    "a[data-test='property-card-link']",
                    "a[href*='homes']",
                    "a[href*='zpid']"
                ]
                
                for selector in link_selectors:
                    try:
                        url_elem = card.find_element(By.CSS_SELECTOR, selector)
                        if url_elem and url_elem.get_attribute("href"):
                            url = url_elem.get_attribute("href")
                            break
                    except:
                        continue
                
                # Create listing dictionary
                listing = {
                    "address": address,
                    "price": price,
                    "beds": beds,
                    "baths": baths,
                    "sqft": sqft,
                    "url": url
                }
                
                listings.append(listing)
                print(f"Extracted listing {i+1}/{len(property_cards)}: {address}")
                
                # Add random delay between processing cards
                if i % 5 == 0:  # Every few cards
                    random_delay(0.3, 0.8)
                
            except Exception as e:
                print(f"Error extracting listing {i+1}: {e}")
                continue
        
    except Exception as e:
        print(f"An error occurred during extraction: {e}")
        driver.save_screenshot("extraction_error.png")
    
    return listings

def navigate_to_next_page(driver, current_page):
    """Navigate to the next page of search results"""
    try:
        print(f"\nAttempting to navigate to page {current_page + 1}...")
        
        # First approach: Try to find and click on the "Next page" button
        next_button_selectors = [
            "a[title='Next page']",
            "a.next",
            "button[aria-label='Next page']",
            "button[title='Next page']",
            ".PaginationJumpItem-c11n-8-109-3__sc-gcj2ze-0.next",
            "a[rel='next']"
        ]
        
        for selector in next_button_selectors:
            try:
                next_button = WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                if next_button and next_button.is_displayed():
                    print(f"Found 'Next page' button using selector: {selector}")
                    
                    # Scroll to the button
                    driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", next_button)
                    random_delay(1, 2)
                    
                    # Click the button
                    next_button.click()
                    print(f"Clicked 'Next page' button, navigating to page {current_page + 1}...")
                    
                    # Wait for the page to load
                    random_delay(5, 8)
                    return True
            except:
                continue
        
        # Second approach: Try to find pagination elements and click on the next page number
        try:
            # Look for the pagination container
            pagination_selectors = [
                ".PaginationControls-c11n-8-109-3__sc-so26y6-0",
                ".zsg-pagination",
                "[data-testid='pagination']",
                "nav[aria-label='pagination']"
            ]
            
            for selector in pagination_selectors:
                try:
                    pagination = driver.find_element(By.CSS_SELECTOR, selector)
                    if pagination:
                        # Find all page number elements
                        page_links = pagination.find_elements(By.TAG_NAME, "a")
                        
                        # Look for the specific page number link
                        for link in page_links:
                            if link.text.strip() == str(current_page + 1):
                                # Found the next page link
                                print(f"Found link to page {current_page + 1}")
                                
                                # Scroll to the link
                                driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", link)
                                random_delay(1, 2)
                                
                                # Click the link
                                link.click()
                                print(f"Clicked link to page {current_page + 1}...")
                                
                                # Wait for the page to load
                                random_delay(5, 8)
                                return True
                except:
                    continue
        except:
            print("Could not find pagination elements")
        
        # Third approach: URL modification
        # This is tricky because Zillow's URL structure can be complex
        # We need to parse the current URL and modify the pagination parameter
        
        current_url = driver.current_url
        print(f"Attempting URL modification from: {current_url}")
        
        # Check if there's a searchQueryState parameter
        if "searchQueryState" in current_url:
            try:
                # Parse the URL
                parsed_url = urllib.parse.urlparse(current_url)
                
                # Get the query parameters
                query_params = urllib.parse.parse_qs(parsed_url.query)
                
                # If searchQueryState exists, it's a JSON string
                if "searchQueryState" in query_params:
                    # Parse the JSON string
                    search_state = json.loads(urllib.parse.unquote(query_params["searchQueryState"][0]))
                    
                    # Check if pagination exists
                    if "pagination" in search_state:
                        # Update the currentPage
                        if "currentPage" in search_state["pagination"]:
                            search_state["pagination"]["currentPage"] = current_page
                        else:
                            search_state["pagination"] = {"currentPage": current_page}
                    else:
                        # Add pagination if it doesn't exist
                        search_state["pagination"] = {"currentPage": current_page}
                    
                    # Update the query parameter
                    query_params["searchQueryState"] = [json.dumps(search_state)]
                    
                    # Rebuild the query string
                    new_query = urllib.parse.urlencode(query_params, doseq=True)
                    
                    # Rebuild the URL
                    new_url = urllib.parse.urlunparse((
                        parsed_url.scheme,
                        parsed_url.netloc,
                        parsed_url.path,
                        parsed_url.params,
                        new_query,
                        parsed_url.fragment
                    ))
                    
                    print(f"Navigating to constructed URL: {new_url}")
                    driver.get(new_url)
                    
                    # Wait for the page to load
                    random_delay(5, 8)
                    return True
            except Exception as e:
                print(f"Error modifying URL: {e}")
        
        # If all approaches fail, try direct URL construction
        try:
            # For simpler URLs, add or update the page parameter
            if "?" in current_url:
                if "page=" in current_url:
                    # Replace the existing page parameter
                    new_url = current_url.replace(f"page={current_page}", f"page={current_page + 1}")
                else:
                    # Add the page parameter
                    new_url = current_url + f"&page={current_page + 1}"
            else:
                # Add the page parameter
                new_url = current_url + f"?page={current_page + 1}"
            
            print(f"Trying simplified URL: {new_url}")
            driver.get(new_url)
            
            # Wait for the page to load
            random_delay(5, 8)
            
            # Check if we have results on this page
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "ul.photo-cards, [data-test='property-card']"))
                )
                print("Found property cards on the new page.")
                return True
            except:
                print("No property cards found on the new page. URL modification likely failed.")
                return False
        except Exception as e:
            print(f"Error with simplified URL modification: {e}")
        
        # If we got here, we couldn't find a way to the next page
        print(f"Could not find a way to navigate to page {current_page + 1}")
        return False
        
    except Exception as e:
        print(f"Error navigating to next page: {e}")
        driver.save_screenshot(f"next_page_error_{current_page}.png")
        return False

def save_to_csv(listings, filename="massachusetts_listings.csv"):
    """Save listings to a CSV file"""
    if not listings:
        print("No listings to save.")
        return
    
    fieldnames = ["address", "price", "beds", "baths", "sqft", "url"]
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for listing in listings:
            writer.writerow(listing)
    
    print(f"Saved {len(listings)} listings to {filename}")

def main():
    """Main function to run the scraper"""
    print("Starting Zillow scraper for Massachusetts listings...")
    
    # Set up the driver
    driver = setup_driver()
    
    # List to store all listings from all pages
    all_listings = []
    
    # Set maximum number of pages to scrape
    max_pages = 10
    
    try:
        # Search for MA properties on Zillow
        search_successful = search_zillow_for_ma(driver)
        
        if search_successful:
            # Process each page up to max_pages
            for current_page in range(1, max_pages + 1):
                print(f"\n--- Processing Page {current_page} of {max_pages} ---\n")
                
                # Extract listings from current page
                page_listings = extract_listings(driver)
                
                # Display sample listings from this page
                if page_listings:
                    print(f"\nExtracted {len(page_listings)} listings from page {current_page}.")
                    
                    # Add these listings to our master list
                    all_listings.extend(page_listings)
                    
                    print(f"Total listings collected so far: {len(all_listings)}")
                else:
                    print(f"No listings found on page {current_page}.")
                
                # Take a screenshot of the current page
                driver.save_screenshot(f"page_{current_page}_results.png")
                
                # Check if we've reached the last page
                if current_page == max_pages:
                    print(f"Reached maximum number of pages ({max_pages}). Stopping.")
                    break
                
                # Navigate to the next page
                next_page_success = navigate_to_next_page(driver, current_page)
                
                # If we couldn't navigate to the next page, we're done
                if not next_page_success:
                    print(f"Could not navigate to page {current_page + 1}. Stopping.")
                    break
                
                # Check for CAPTCHA after page navigation
                try_solve_captcha(driver)
                
                # Wait for page to load and add some random delays
                random_delay(2, 4)
            
            # After processing all pages, show summary and save results
            if all_listings:
                print(f"\nSuccessfully extracted {len(all_listings)} total listings across all pages.")
                
                # Display sample of final results
                print("\nSample listings:")
                for i, listing in enumerate(all_listings[:5]):
                    print(f"\nListing {i+1}:")
                    for key, value in listing.items():
                        print(f"  {key}: {value}")
                    print("-" * 50)
                
                # Save all listings to CSV
                save_to_csv(all_listings)
            else:
                print("No listings were extracted across all pages.")
        else:
            print("Failed to search for Massachusetts properties on Zillow.")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        try:
            driver.save_screenshot("final_error.png")
        except:
            pass
    
    finally:
        # Close the browser
        try:
            driver.quit()
        except:
            pass  # Ignore errors during driver cleanup

if __name__ == "__main__":
    main()
