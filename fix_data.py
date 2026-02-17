"""
Fix data files across all phases to ensure consistency with code expectations.

Issues being fixed:
1. inventory.csv in phases 2/3/5/6 missing 'condition' and 'category' columns
2. complaints.csv in phases 3/5/6 missing 'vehicle' column
3. faq.json in phases 2/3/5/6 missing 'category' field
"""
import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).parent

# ============================================================================
# 1. Fix inventory.csv - add 'condition' and 'category' columns
# ============================================================================

# Model -> category mapping
CATEGORY_MAP = {
    "Camry": "sedan",
    "CR-V": "suv",
    "F-150": "truck",
    "Equinox": "suv",
    "330i": "sedan",
    "Tucson": "suv",
    "RAV4 Hybrid": "suv",
    "Bronco Sport": "suv",
    "Civic": "sedan",
    "Silverado 1500": "truck",
    "Telluride": "suv",
    "Altima": "sedan",
    "Outback": "suv",
    "Grand Cherokee": "suv",
    "Corolla": "sedan",
    "Ioniq 5": "ev",
    "Mustang Mach-E": "ev",
    "Bolt EUV": "ev",
    "X3": "suv",
    "Sportage": "suv",
    "Rogue": "suv",
    "Crosstrek": "suv",
    "Wrangler": "suv",
    "Tacoma": "truck",
    "Accord": "sedan",
    "Explorer": "suv",
    "Palisade": "suv",
    "EV6": "ev",
    "Frontier": "truck",
    "Highlander Hybrid": "suv",
}


def get_category(model):
    """Determine vehicle category from model name."""
    for key, cat in CATEGORY_MAP.items():
        if key.lower() in model.lower():
            return cat
    return "other"


def get_condition(mileage):
    """Determine condition from mileage."""
    try:
        m = int(mileage)
        return "New" if m < 100 else "Certified Pre-Owned"
    except (ValueError, TypeError):
        return "Unknown"


def fix_inventory_csv(phase_dir):
    """Add 'condition' and 'category' columns if missing."""
    csv_path = phase_dir / "data" / "inventory.csv"
    if not csv_path.exists():
        return

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if "condition" in fieldnames and "category" in fieldnames:
        print(f"  {phase_dir.name}: inventory.csv already has condition + category")
        return

    # Build new fieldnames: insert condition after mileage, category after color
    new_fieldnames = []
    for fn in fieldnames:
        new_fieldnames.append(fn)
        if fn == "mileage" and "condition" not in fieldnames:
            new_fieldnames.append("condition")
        if fn == "color" and "category" not in fieldnames:
            new_fieldnames.append("category")

    # Add data to rows
    for row in rows:
        if "condition" not in row:
            row["condition"] = get_condition(row.get("mileage", "0"))
        if "category" not in row:
            row["category"] = get_category(row.get("model", ""))

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  {phase_dir.name}: Fixed inventory.csv (added condition + category)")


# ============================================================================
# 2. Fix complaints.csv - add 'vehicle' column
# ============================================================================

# Map complaint IDs to vehicle names based on description content
VEHICLE_MAP = {
    "1": "2024 Toyota Camry LE",
    "2": "2023 Ford F-150 XLT",
    "3": "2023 BMW 330i xDrive",
    "4": "2024 Hyundai Tucson SEL",
    "5": "2022 Toyota RAV4 Hybrid XLE",
    "6": "2024 Chevrolet Equinox RS",
    "7": "2020 Honda Accord",
    "8": "2024 Kia Sportage",
    "9": "2023 Ford F-150 XLT",
    "10": "N/A",
    "11": "2024 Honda CR-V EX-L",
    "12": "2024 Toyota Tacoma TRD",
    "13": "2024 Jeep Wrangler 4xe",
    "14": "N/A",
    "15": "2024 Subaru Crosstrek Premium",
}


def fix_complaints_csv(phase_dir):
    """Add 'vehicle' column if missing."""
    csv_path = phase_dir / "data" / "complaints.csv"
    if not csv_path.exists():
        return

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if "vehicle" in fieldnames:
        print(f"  {phase_dir.name}: complaints.csv already has 'vehicle'")
        return

    # Insert 'vehicle' after 'customer_name'
    new_fieldnames = []
    for fn in fieldnames:
        new_fieldnames.append(fn)
        if fn == "customer_name":
            new_fieldnames.append("vehicle")

    for row in rows:
        complaint_id = row.get("id", "")
        row["vehicle"] = VEHICLE_MAP.get(str(complaint_id), "N/A")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  {phase_dir.name}: Fixed complaints.csv (added 'vehicle' column)")


# ============================================================================
# 3. Fix faq.json - add 'category' field
# ============================================================================

# Keyword-based category assignment
FAQ_CATEGORIES = {
    "financing": ["financing", "finance", "APR", "loan", "credit", "payment", "pre-approved"],
    "service": ["service", "oil change", "maintenance", "loaner", "warranty claim"],
    "sales": ["test drive", "buy", "purchase", "documents", "delivery", "hours", "sales"],
    "inventory": ["electric", "EV", "charging", "stock"],
    "policies": ["return", "trade-in", "warranty", "certified pre-owned", "gap insurance"],
    "promotions": ["promotion", "deals", "referral"],
}


def categorize_faq(question, answer):
    """Assign a category based on FAQ content."""
    text = (question + " " + answer).lower()
    for category, keywords in FAQ_CATEGORIES.items():
        for kw in keywords:
            if kw.lower() in text:
                return category
    return "general"


def fix_faq_json(phase_dir):
    """Add 'category' field if missing."""
    json_path = phase_dir / "data" / "faq.json"
    if not json_path.exists():
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    if data and "category" in data[0]:
        print(f"  {phase_dir.name}: faq.json already has 'category'")
        return

    for entry in data:
        if "category" not in entry:
            entry["category"] = categorize_faq(entry["question"], entry["answer"])

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    print(f"  {phase_dir.name}: Fixed faq.json (added 'category' field)")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("Fixing data files across all phases...\n")

    for phase_dir in sorted(ROOT.glob("phase-*")):
        print(f"\n{phase_dir.name}:")
        fix_inventory_csv(phase_dir)
        fix_complaints_csv(phase_dir)
        fix_faq_json(phase_dir)

    print("\n\nDone! All data files have been updated.")
