from bs4 import BeautifulSoup
import time

def get_component_ids_from_category(t_value, driver):
    """
    Scrape component IDs from a category page (given a t-value) using a live Selenium driver.
    """
    url = f"https://www.bauteileditor.de/elements/list/?t={t_value}"
    driver.get(url)
    time.sleep(2)

    prev_ids = set()
    max_scrolls = 30
    for i in range(max_scrolls):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2.5)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        cards = soup.select("div[id^='elca-sheet-']")
        current_ids = set(int(card['id'].split("-")[-1]) for card in cards)
        if current_ids == prev_ids:
            break
        prev_ids = current_ids

    return sorted(prev_ids)
