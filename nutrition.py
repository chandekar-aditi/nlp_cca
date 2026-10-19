
import pandas as pd


def _clean(value):
    if value is None:
        return "Not listed"

    try:
        if pd.isna(value):
            return "Not listed"
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return "Not listed"

    return text


def _get(product, column):
    return _clean(product.get(column, "Not listed"))


def nutrition_table(product):
    """Format nutrition values using the dataset's recorded values."""
    fields = [
        ("Serving size", "Serving_Size_g", "g"),
        ("Calories", "Calories_kcal", "kcal"),
        ("Carbohydrates", "Carbohydrates_g", "g"),
        ("Protein", "Proteins_g", "g"),
        ("Total fat", "Total_Fat_g", "g"),
        ("Saturated fat", "Saturated_Fat_g", "g"),
        ("Trans fat", "Trans_Fat_g", "g"),
        ("Sugar", "Sugar_g", "g"),
        ("Sodium", "Sodium_mg", "mg"),
        ("Dietary fibre", "Dietary_Fiber_g", "g"),
        ("Cholesterol", "Cholesterol_mg", "mg"),
        ("Calcium", "Calcium_mg", "mg"),
        ("Iron", "Iron_mg", "mg"),
    ]

    lines = ["| Nutrient | Recorded value |", "|---|---|"]

    for label, column, unit in fields:
        value = _get(product, column)
        if value == "Not listed":
            display = value
        else:
            display = f"{value} {unit}"

        lines.append(f"| {label} | {display} |")

    lines.append("")
    lines.append(
        "Values are shown as recorded in the dataset. "
        "Check the product label for the stated serving basis."
    )

    return "\n".join(lines)


def additive_categories(product, intent="categories"):
    """Return only the relevant ingredient or additive information."""
    if intent == "allergens":
        return (
            f"**Allergens:** {_get(product, 'Allergens')}\n\n"
            "This reflects the database entry; check the package label "
            "for the latest allergen and cross-contact warnings."
        )

    if intent == "ingredients":
        return (
            f"**Ingredients:** {_get(product, 'Ingredients')}\n\n"
            f"**Ingredient explanation:** "
            f"{_get(product, 'Ingredient_Explanation')}"
        )

    q = str(intent).lower()

    # Check the actual ingredient list first. A secondary classification
    # column marked "Not listed" does not override explicit ingredients.
    if "palm oil" in q:
        ingredients = _get(product, "Ingredients")
        palm_field = _get(product, "Added palm oil")

        if ingredients != "Not listed" and "palm" in ingredients.lower():
            return (
                "**Palm oil:** Yes, the ingredient list explicitly mentions "
                f"it.\n\n**Ingredient list:** {ingredients}"
            )

        if palm_field != "Not listed":
            return f"**Added palm oil:** {palm_field}"

        if ingredients != "Not listed":
            return (
                "The ingredient list available in the database does not "
                "explicitly mention palm oil. This is not a guarantee that "
                "the product is palm-oil-free; verify the current package."
            )

        return "Palm oil information is not listed in the available database."

    if "sugar" in q:
        return (
            f"**Added sugar classification:** {_get(product, 'Added sugar')}\n\n"
            f"**Sugar per 100 g (as recorded):** {_get(product, 'Sugar_g')} g"
        )

    category_columns = [
        "Emulsifiers",
        "Acidity regulators",
        "Preservatives",
        "Colours",
        "Stabilizers",
        "Thickeners",
        "Raising/leavening agents",
        "Humectants",
        "Antioxidants",
        "Flavour enhancers",
        "Glazing agents",
        "Anticaking agents",
        "Gelling agents",
        "Flour-treatment agents and improvers",
        "Other functional ingredients",
        "Sweeteners",
        "Flavouring substances",
    ]

    lines = ["| Category | Database entry |", "|---|---|"]

    for column in category_columns:
        value = _get(product, column)
        if value != "Not listed":
            lines.append(f"| {column} | {value} |")

    if len(lines) == 2:
        return "No additive-category entries are listed for this product."

    return "\n".join(lines)


def compare_products(product1, product2):
    """Compare the exact two supplied product rows."""
    name1 = _get(product1, "Item name")
    name2 = _get(product2, "Item name")

    fields = [
        ("Brand", "Brand_Name"),
        ("Category", "Category"),
        ("Ingredients", "Ingredients"),
        ("Allergens", "Allergens"),
        ("Calories (kcal)", "Calories_kcal"),
        ("Carbohydrates (g)", "Carbohydrates_g"),
        ("Protein (g)", "Proteins_g"),
        ("Total fat (g)", "Total_Fat_g"),
        ("Saturated fat (g)", "Saturated_Fat_g"),
        ("Trans fat (g)", "Trans_Fat_g"),
        ("Sugar (g)", "Sugar_g"),
        ("Sodium (mg)", "Sodium_mg"),
        ("Price (INR)", "Price_INR"),
        ("Added palm oil", "Added palm oil"),
        ("Added sugar", "Added sugar"),
    ]

    lines = [
        f"| Detail | {name1} | {name2} |",
        "|---|---|---|",
    ]

    for label, column in fields:
        lines.append(
            f"| {label} | {_get(product1, column)} | "
            f"{_get(product2, column)} |"
        )

    lines.extend([
        "",
        "Values are reproduced from the database. "
        "'Not listed' means the database has no usable entry; "
        "it does not mean the ingredient or nutrient is absent.",
    ])

    return "\n".join(lines)
