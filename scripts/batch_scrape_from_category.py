import pandas as pd
from pollutant_predictor.scraping.get_component_ids import get_component_ids_from_category
from pollutant_predictor.scraping.scrape_materials import setup_driver, scrape_component
from pollutant_predictor.config.paths import RAW_DIR


# Categories to scrape (t-values)
t_values = [237, 246, 256, 264, 269]

if __name__ == "__main__":
    driver = setup_driver()
    all_data = []

    for t in t_values:
        print(f"🔍 Getting component IDs for category t={t}")
        component_ids = get_component_ids_from_category(t, driver)

        for i, comp_id in enumerate(component_ids):
            print(f"  → [{i+1}/{len(component_ids)}] Scraping component {comp_id}...")
            df = scrape_component(driver, comp_id)
            if not df.empty:
                df["t_value"] = t
                out_path = RAW_DIR / f"{comp_id}_materials.csv"
                df.to_csv(out_path, index=False)
                all_data.append(df)

    driver.quit()

    # Save full combined CSV
    if all_data:
        full_df = pd.concat(all_data, ignore_index=True)
        combined_path = RAW_DIR / "all_scraped_materials.csv"
        full_df.to_csv(combined_path, index=False)
        print(f"\n✅ Done. Combined file saved to: {combined_path}")
    else:
        print("⚠️ No materials were scraped.")
