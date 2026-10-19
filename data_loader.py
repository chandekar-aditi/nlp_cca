
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def load_data():
    products_path = DATA_DIR / "products.csv"
    additives_path = DATA_DIR / "additives.csv"

    if not products_path.exists():
        raise FileNotFoundError(
            "Missing data/products.csv"
        )

    if not additives_path.exists():
        raise FileNotFoundError(
            "Missing data/additives.csv"
        )

    products = pd.read_csv(products_path).fillna("")
    additives = pd.read_csv(additives_path).fillna("")

    numeric_columns = [
        "Serving_Size_g",
        "Calories_kcal",
        "Carbohydrates_g",
        "Proteins_g",
        "Total_Fat_g",
        "Saturated_Fat_g",
        "Trans_Fat_g",
        "Sugar_g",
        "Sodium_mg",
        "Dietary_Fiber_g",
        "Cholesterol_mg",
        "Calcium_mg",
        "Iron_mg",
        "Price_INR",
    ]

    for column in numeric_columns:
        if column in products.columns:
            products[column] = pd.to_numeric(
                products[column],
                errors="coerce"
            )

    return products, additives
