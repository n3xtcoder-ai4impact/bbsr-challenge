import time
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

from pollutant_predictor.config.config import SESSION_COOKIE, CHROMEDRIVER_PATH, BASE_URL


def setup_driver():
    options = Options()
    # options.add_argument("--headless=new")  # uncomment for silent mode
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_experimental_option("detach", True)

    driver = webdriver.Chrome(service=Service(CHROMEDRIVER_PATH), options=options)
    driver.get(BASE_URL + "/favicon.ico")  # initialize session
    driver.delete_all_cookies()
    driver.add_cookie({
        "name": "sid",
        "value": SESSION_COOKIE,
        "domain": "www.bauteileditor.de",
        "path": "/",
    })
    return driver


def get_l1_subcomponents(driver, component_id):
    driver.get(f"{BASE_URL}/element?uuid={component_id}")
    time.sleep(2)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    items = soup.select("ul#subcomponents li a")
    return [(item.text.strip(), item["href"].split("=")[-1]) for item in items]


def get_materials_from_subcomponent(driver, sub_id):
    driver.get(f"{BASE_URL}/element?uuid={sub_id}")
    time.sleep(2)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    materials = []

    for li in soup.select("ol.sortable > li"):
        name = li.select_one(".layer-title")
        uuid = li.get("data-uuid")
        thickness = li.select_one(".thickness .value")
        materials.append({
            "material_name": name.text.strip() if name else None,
            "uuid": uuid,
            "thickness": thickness.text.strip() if thickness else None,
            "subcomponent_id": sub_id
        })

    return materials


def scrape_component(driver, component_id):
    try:
        subcomponents = get_l1_subcomponents(driver, component_id)
        all_materials = []
        for _, sub_id in subcomponents:
            all_materials.extend(get_materials_from_subcomponent(driver, sub_id))
        df = pd.DataFrame(all_materials)
        df["component_id"] = component_id
        return df
    except Exception as e:
        print(f"❌ Failed to scrape component {component_id}: {e}")
        return pd.DataFrame()

