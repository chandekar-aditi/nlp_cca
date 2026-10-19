
import pandas as pd

from product_matcher import (
    normalize,
    find_product,
    find_two_products,
    detect_intent,
)
from nutrition import additive_categories, compare_products


def sample_products():
    return pd.DataFrame([
        {
            "Item name": "Parle-G Original Gluco Biscuits",
            "Brand_Name": "Parle-G",
            "Ingredients": "Wheat flour, sugar, refined palm oil",
            "Added palm oil": "Palm oil/palmolein",
            "Added sugar": "Sugar",
            "Allergens": "Wheat; Milk",
            "Sugar_g": 25.5,
            "Sodium_mg": 296,
            "Emulsifiers": "INS 472e",
        },
        {
            "Item name": "Britannia Little Hearts Classic Crunch Biscuits",
            "Brand_Name": "Britannia",
            "Ingredients": "Refined wheat flour, sugar, edible vegetable oils (palm), emulsifier (322)",
            "Added palm oil": "Not listed",
            "Added sugar": "Sugar",
            "Allergens": "Wheat; Milk",
            "Sugar_g": 24.0,
            "Sodium_mg": 309,
            "Emulsifiers": "INS 322 / lecithin",
        },
        {
            "Item name": "Britannia Good Day Cashew Cookies Family Pack",
            "Brand_Name": "Britannia",
            "Ingredients": "Wheat flour, sugar, palm oil, cashew",
            "Added palm oil": "Not listed",
            "Added sugar": "Sugar; Invert sugar/syrup",
            "Allergens": "Wheat; Milk; Cashew",
            "Sugar_g": 22.0,
            "Sodium_mg": 0.0,
            "Emulsifiers": "INS 322 / lecithin; INS 471; INS 472e",
        },
    ])


def test_normalize_nbsp_and_punctuation():
    assert normalize("Little\u00a0Hearts!") == "little hearts"


def test_find_parle_g():
    products = sample_products()
    row, score = find_product("Does Parle-G contain palm oil?", products)
    assert row is not None
    assert row["Item name"] == "Parle-G Original Gluco Biscuits"


def test_find_shortened_little_hearts():
    products = sample_products()
    row, score = find_product("Little Hearts", products)
    assert row is not None
    assert "Little Hearts" in row["Item name"]


def test_unknown_product_does_not_pick_random_row():
    products = sample_products()
    row, score = find_product("What about a dragonfruit smoothie?", products)
    assert row is None


def test_detect_ins_322_as_additive():
    assert detect_intent("What is INS 322?") == "additive"


def test_detect_palm_oil_question():
    assert detect_intent("Does this contain palm oil?") == "categories"


def test_palm_oil_ingredient_list_overrides_not_listed():
    products = sample_products()
    row = products.iloc[1]
    answer = additive_categories(row, "palm oil")
    assert "Yes" in answer
    assert "palm" in answer.lower()


def test_compare_exact_two_products():
    products = sample_products()
    answer = compare_products(products.iloc[0], products.iloc[1])
    assert "Parle-G Original Gluco Biscuits" in answer
    assert "Britannia Little Hearts Classic Crunch Biscuits" in answer


def test_find_two_products_in_comparison():
    products = sample_products()
    pair = find_two_products(
        "Compare Parle-G Original Gluco Biscuits with Little Hearts",
        products,
    )
    assert len(pair) == 2
    names = [row["Item name"] for row in pair]
    assert "Parle-G Original Gluco Biscuits" in names
    assert any("Little Hearts" in name for name in names)


if __name__ == "__main__":
    tests = [
        test_normalize_nbsp_and_punctuation,
        test_find_parle_g,
        test_find_shortened_little_hearts,
        test_unknown_product_does_not_pick_random_row,
        test_detect_ins_322_as_additive,
        test_detect_palm_oil_question,
        test_palm_oil_ingredient_list_overrides_not_listed,
        test_compare_exact_two_products,
        test_find_two_products_in_comparison,
    ]

    for test in tests:
        test()
        print(f"PASS: {test.__name__}")

    print(f"\n{len(tests)} tests passed.")
