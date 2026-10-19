
import re
from difflib import SequenceMatcher

import pandas as pd
import streamlit as st

from data_loader import load_data
from retriever import build_retriever, retrieve_context
from llm import answer_with_groq
from nutrition import nutrition_table, additive_categories, compare_products


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="IngredientIQ",
    page_icon="🥣",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM DESIGN — CREAM, SAGE, FOREST GREEN, ORANGE
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

    :root {
        --forest: #14563c;
        --green: #36784e;
        --leaf: #68a83f;
        --orange: #e75b28;
        --cream: #fbf8ef;
        --sage: #edf2e6;
        --line: #e5e8dd;
        --ink: #1c3444;
    }

    .stApp {
        background: var(--cream);
        font-family: 'DM Sans', sans-serif;
        color: var(--ink);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1500px;
        padding: 1.1rem 1.4rem 1.5rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #eff4e8 0%, #e7efdf 100%);
        border-right: 1px solid #e0e8d8;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.3rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 4px 0 24px;
    }

    .brand-icon {
        font-size: 34px;
        background: #dbe9d5;
        border-radius: 18px;
        padding: 9px;
    }

    .brand-name {
        font-size: 25px;
        font-weight: 800;
        letter-spacing: -1px;
        color: var(--forest);
        line-height: 1.1;
    }

    .brand-name span {
        color: var(--orange);
    }

    .brand-tagline {
        font-size: 12px;
        margin-top: 4px;
        color: #476254;
    }

    .hero {
        padding: 5px 10px 18px;
        border-bottom: 1px solid #e9e5d9;
        margin-bottom: 15px;
    }

    .hero-title {
        font-size: clamp(36px, 4.3vw, 60px);
        line-height: 1.02;
        letter-spacing: -2.5px;
        font-weight: 800;
        color: var(--forest);
        margin: 0;
    }

    .hero-title span {
        color: var(--orange);
    }

    .hero-subtitle {
        font-size: 19px;
        color: #253d42;
        margin: 10px 0 4px;
    }

    .hero-description {
        font-size: 15px;
        color: #526067;
        margin-bottom: 16px;
    }

    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }

    .pill {
        border: 1px solid #ece8de;
        background: rgba(255,255,255,.88);
        border-radius: 30px;
        padding: 9px 14px;
        font-size: 13px;
        color: #243c3d;
        box-shadow: 0 2px 6px rgba(35,60,40,.04);
    }

    .section-heading {
        color: #173e32;
        font-weight: 700;
        font-size: 15px;
        margin: 10px 0 12px;
    }

    div[data-testid="stChatMessage"] {
        background: rgba(255,255,255,.91);
        border: 1px solid #e3e9e1;
        border-radius: 17px;
        padding: 14px 17px;
        margin-bottom: 12px;
        box-shadow: 0 2px 8px rgba(25,60,40,.025);
    }

    div[data-testid="stChatInput"] {
        background: white;
        border: 1px solid #e7e4da;
        border-radius: 28px;
        box-shadow: 0 3px 12px rgba(20,50,30,.04);
    }

    div[data-testid="stChatInput"] textarea {
        font-family: 'DM Sans', sans-serif;
    }

    div.stButton > button {
        border-radius: 13px;
        border: 1px solid #e1e8dc;
        background: rgba(255,255,255,.78);
        color: #203d3c;
        min-height: 43px;
        font-weight: 500;
        text-align: left;
        transition: .15s ease;
    }

    div.stButton > button:hover {
        border-color: #75a57c;
        color: var(--forest);
        background: white;
    }

    div.stButton > button[kind="primary"] {
        background: var(--green);
        border-color: var(--green);
        color: white;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 14px;
    }

    .welcome-card {
        background: rgba(255,255,255,.68);
        border: 1px solid #e8e9df;
        border-radius: 18px;
        padding: 20px;
        margin: 10px 0 16px;
    }

    .welcome-card h3 {
        color: var(--forest);
        margin-top: 0;
    }

    .small-note {
        font-size: 12px;
        color: #65756a;
        line-height: 1.5;
    }

    hr {
        border-color: #e3e7dc;
    }

    @media (max-width: 700px) {
        .block-container {
            padding: .7rem .6rem;
        }

        .hero-title {
            font-size: 38px;
        }

        .hero-subtitle {
            font-size: 16px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD DATABASE
# =========================================================

products, additives = load_data()

if products is None or products.empty:
    st.error(
        "The product database is empty. Check your CSV path in data_loader.py."
    )
    st.stop()

with st.spinner("Preparing IngredientIQ..."):
    retriever = build_retriever(products, additives)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_product_name" not in st.session_state:
    st.session_state.active_product_name = None

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# =========================================================
# GENERAL HELPERS
# =========================================================

def clean_value(value):
    if value is None:
        return "Not listed"

    try:
        if pd.isna(value):
            return "Not listed"
    except (TypeError, ValueError):
        pass

    value = str(value).strip()

    if not value or value.lower() in {"nan", "none", "null"}:
        return "Not listed"

    return value


def normalize(text):
    text = str(text or "").replace("\u00a0", " ").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def product_name(product):
    if product is None:
        return None
    return clean_value(product.get("Item name", "Not listed"))


def get_active_product():
    saved_name = st.session_state.get("active_product_name")

    if not saved_name:
        return None

    target = normalize(saved_name)

    for _, row in products.iterrows():
        if normalize(row.get("Item name", "")) == target:
            return row

    return None


def set_active_product(product):
    if product is not None:
        name = product_name(product)

        if name and name != "Not listed":
            st.session_state.active_product_name = name


def get_number(product, columns):
    """Return the first valid numeric value found in the product row."""
    for column in columns:
        if column not in product.index:
            continue

        value = pd.to_numeric(product.get(column), errors="coerce")

        if pd.notna(value) and value >= 0:
            return float(value)

    return None


def format_number(value):
    return "Not listed" if value is None else f"{value:g}"


# =========================================================
# PRODUCT MATCHING — BUILT INTO THIS FILE
# =========================================================

GENERIC_WORDS = {
    "the", "a", "an", "and", "or", "with", "of", "for",
    "pack", "family", "biscuits", "biscuit", "cookies",
    "cookie", "classic", "crunch", "original", "gluco",
    "plain", "product",
}


def product_aliases(row):
    """
    Create product aliases without relying on generic brand-only matches.
    Examples include 'little hearts', 'good day cashew', 'marie gold',
    and 'parle g'.
    """
    aliases = set()

    item_name = normalize(row.get("Item name", ""))
    brand = normalize(row.get("Brand_Name", ""))

    if item_name:
        aliases.add(item_name)
        tokens = item_name.split()

        for start in range(len(tokens)):
            for end in range(start + 2, min(len(tokens), start + 4) + 1):
                phrase_tokens = tokens[start:end]

                distinctive = [
                    token for token in phrase_tokens
                    if token not in GENERIC_WORDS
                ]

                if len(distinctive) >= 2:
                    aliases.add(" ".join(phrase_tokens))

        distinctive_tokens = [
            token for token in tokens
            if token not in GENERIC_WORDS
        ]

        if len(distinctive_tokens) >= 2:
            aliases.add(" ".join(distinctive_tokens))

    # Parle-G is a distinctive brand/product alias.
    # Britannia alone is deliberately not accepted as a product name.
    if brand and brand != "britannia":
        aliases.add(brand)

    return {alias for alias in aliases if alias}


def contains_phrase(text, phrase):
    return bool(
        phrase and re.search(
            rf"(?<!\w){re.escape(phrase)}(?!\w)",
            text,
        )
    )


def find_explicit_product(question):
    """Resolve a named product from the full question."""
    q = normalize(question)

    if not q:
        return None

    candidates = []

    for _, row in products.iterrows():
        aliases = product_aliases(row)

        matches = [
            alias for alias in aliases
            if contains_phrase(q, alias)
        ]

        if matches:
            best_alias = max(matches, key=len)
            candidates.append((len(best_alias), row))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_length, best_row = candidates[0]
    best_name = normalize(best_row.get("Item name", ""))

    # Do not guess if different products match equally well.
    for length, row in candidates[1:]:
        if (
            length == best_length
            and normalize(row.get("Item name", "")) != best_name
        ):
            return None

    return best_row


def split_comparison(question):
    """Split a comparison into two independently resolved product mentions."""
    raw = str(question or "").replace("\u00a0", " ").strip()

    raw = re.sub(
        r"^\s*(compare|comparison of|difference between)\s+",
        "",
        raw,
        flags=re.IGNORECASE,
    )

    separators = [
        r"\s+versus\s+",
        r"\s+vs\.?\s+",
        r"\s+compared\s+with\s+",
        r"\s+and\s+",
        r"\s+with\s+",
    ]

    for separator in separators:
        parts = re.split(
            separator, raw, maxsplit=1, flags=re.IGNORECASE
        )

        if len(parts) == 2 and all(part.strip() for part in parts):
            return parts[0].strip(), parts[1].strip()

    return None, None


def resolve_comparison_side(side):
    """Resolve one side only; never substitute a different product."""
    side_normalized = normalize(side)

    if not side_normalized:
        return None

    matches = []

    for _, row in products.iterrows():
        aliases = product_aliases(row)

        found = [
            alias for alias in aliases
            if contains_phrase(side_normalized, alias)
        ]

        if found:
            matches.append((max(len(alias) for alias in found), row))

    if not matches:
        return None

    matches.sort(key=lambda item: item[0], reverse=True)
    best_length, best_row = matches[0]
    best_name = normalize(best_row.get("Item name", ""))

    for length, row in matches[1:]:
        if (
            length == best_length
            and normalize(row.get("Item name", "")) != best_name
        ):
            return None

    return best_row


def find_two_products(question):
    """Return the two requested products in the order they were mentioned."""
    left, right = split_comparison(question)

    if not left or not right:
        return []

    product1 = resolve_comparison_side(left)
    product2 = resolve_comparison_side(right)

    if product1 is None or product2 is None:
        return []

    name1 = normalize(product1.get("Item name", ""))
    name2 = normalize(product2.get("Item name", ""))

    if not name1 or not name2 or name1 == name2:
        return []

    return [product1, product2]


# =========================================================
# QUESTION INTENT
# =========================================================

def detect_intent(question):
    q = normalize(question)

    if (
        re.search(r"\bcompare\b", q)
        or re.search(r"\bcomparison\b", q)
        or re.search(r"\bversus\b", q)
        or re.search(r"\bvs\b", q)
        or re.search(r"\bdifference between\b", q)
        or re.search(r"\bwhich is better\b", q)
    ):
        return "compare"

    if (
        q in {"hi", "hello", "hey", "hii", "heyy"}
        or any(x in q for x in [
            "good morning", "good afternoon", "good evening",
            "how are you", "who are you", "what can you do",
            "help me get started",
        ])
    ):
        return "greeting"

    if any(x in q for x in [
        "allergen", "allergens", "allergy", "allergies",
        "gluten", "milk allergy", "contains nuts",
    ]):
        return "allergens"

    if any(x in q for x in [
        "nutrition", "nutritional", "calorie", "calories",
        "protein", "sugar content", "sodium", "fat content",
        "nutrient", "nutrition facts", "how much sugar",
        "how much fat", "how many calories", "how much protein",
    ]):
        return "nutrition"

    if any(x in q for x in [
        "preservative", "preservatives", "palm oil",
        "added sugar", "colour", "color", "stabilizer",
        "stabilizers", "emulsifier", "emulsifiers",
        "what additives", "which additives", "additive categories",
        "raising agent", "raising agents", "thickener", "sweetener",
    ]):
        return "categories"

    if (
        re.search(r"\bins?\s*-?\s*\d+", q)
        or any(x in q for x in [
            "fssai", "additive", "ins code", "what does ins", "what is ins",
        ])
    ):
        return "additive"

    if any(x in q for x in [
        "ingredient", "ingredients", "made of", "made from",
        "what is in", "what s in", "composition", "break down",
        "explain them", "explain those", "explain these",
        "what goes into",
    ]):
        return "ingredients"

    return "general"


# =========================================================
# INGREDIENT HELPERS
# =========================================================

def ingredient_list_only(product):
    ingredients = clean_value(product.get("Ingredients"))

    if ingredients == "Not listed":
        return (
            f"The ingredient list for **{product_name(product)}** "
            "is not available in the database."
        )

    return (
        f"### Ingredients in {product_name(product)}\n\n"
        f"{ingredients}"
    )


def explanation_context(product):
    return (
        f"Product name: {product_name(product)}\n"
        f"Brand: {clean_value(product.get('Brand_Name'))}\n\n"
        f"Ingredients (from CSV):\n"
        f"{clean_value(product.get('Ingredients'))}\n\n"
        f"Ingredient_Explanation (from CSV):\n"
        f"{clean_value(product.get('Ingredient_Explanation'))}\n\n"
        f"Allergens (from CSV):\n"
        f"{clean_value(product.get('Allergens'))}\n"
    )


def is_ingredient_list_request(question):
    q = normalize(question)

    list_phrases = [
        "what are the ingredients",
        "list the ingredients",
        "show the ingredients",
        "ingredient list",
        "what ingredients are in",
        "what are its ingredients",
        "what are the ingredients in",
        "what is it made of",
        "what is this made of",
        "what goes into",
        "what s inside",
        "whats inside",
        "what is inside",
    ]

    explanation_phrases = [
        "explain", "meaning", "what does", "what do",
        "what each", "function", "purpose", "simply",
        "in simple words", "in simple english", "break down",
    ]

    if any(phrase in q for phrase in explanation_phrases):
        return False

    return any(phrase in q for phrase in list_phrases)


def is_ingredient_explanation_request(question):
    q = normalize(question)

    return any(phrase in q for phrase in [
        "explain the ingredients",
        "explain those ingredients",
        "explain these ingredients",
        "explain it simply",
        "explain them simply",
        "what do those ingredients do",
        "what do these ingredients do",
        "what does each ingredient do",
        "explain each ingredient",
        "break down the ingredients",
        "ingredients in simple words",
        "ingredients in simple english",
        "what do the ingredients mean",
        "can you explain those",
        "can you explain these",
    ])


def is_palm_oil_question(question):
    q = normalize(question)
    return "palm oil" in q or "palmolein" in q


def is_allergen_question(question):
    q = normalize(question)

    return any(word in q for word in [
        "allergen", "allergens", "allergy", "allergies",
        "gluten", "milk allergy", "nut allergy",
    ])


def is_additive_question(question):
    q = normalize(question)

    return bool(
        re.search(r"\bins?\s*-?\s*\d+\s*[a-z]?\b", q)
        or any(word in q for word in [
            "fssai", "additive", "emulsifier", "preservative",
            "colour", "color",
        ])
    )


def is_nutrition_question(question):
    q = normalize(question)

    return any(word in q for word in [
        "nutrition", "nutritional", "calorie", "calories",
        "sugar", "protein", "sodium", "fat", "carbohydrate",
        "carbs", "fibre", "fiber", "cholesterol", "calcium",
        "iron", "serving size",
    ])


# =========================================================
# NUTRITION ANSWERS
# =========================================================

def nutrition_answer(question, product):
    q = normalize(question)

    fields = [
        (["saturated fat"], "Saturated_Fat_g", "g"),
        (["trans fat"], "Trans_Fat_g", "g"),
        (["total fat", "fat content", "how much fat"], "Total_Fat_g", "g"),
        (["calorie", "calories"], "Calories_kcal", "kcal"),
        (["protein"], "Proteins_g", "g"),
        (["sugar", "sweetness"], "Sugar_g", "g"),
        (["sodium"], "Sodium_mg", "mg"),
        (["carbohydrate", "carbs"], "Carbohydrates_g", "g"),
        (["fibre", "fiber"], "Dietary_Fiber_g", "g"),
        (["cholesterol"], "Cholesterol_mg", "mg"),
        (["calcium"], "Calcium_mg", "mg"),
        (["iron"], "Iron_mg", "mg"),
    ]

    for keywords, column, unit in fields:
        if any(keyword in q for keyword in keywords):
            value = clean_value(product.get(column))

            if value == "Not listed":
                return (
                    f"The {column.replace('_', ' ').lower()} value for "
                    f"**{product_name(product)}** is not listed in the database."
                )

            return (
                f"**{product_name(product)}** has **{value} {unit}** "
                f"recorded for {column.replace('_', ' ').lower()} in the "
                "database. Check the package for the stated serving basis."
            )

    return (
        f"### Nutrition for {product_name(product)}\n\n"
        f"{nutrition_table(product)}"
    )


# =========================================================
# COMPARISON SUMMARY
# =========================================================

def comparison_summary(product1, product2):
    name1 = product_name(product1)
    name2 = product_name(product2)

    sugar1 = get_number(product1, [
        "Sugar_g", "Sugars_g", "Sugars (g)", "Sugar (g)"
    ])
    sugar2 = get_number(product2, [
        "Sugar_g", "Sugars_g", "Sugars (g)", "Sugar (g)"
    ])

    sodium1 = get_number(product1, ["Sodium_mg", "Sodium (mg)"])
    sodium2 = get_number(product2, ["Sodium_mg", "Sodium (mg)"])

    st.markdown("---")
    st.markdown("### 🏆 Which is better?")
    st.caption(
        "A quick comparison of the recorded nutrition data. "
        "Lower sugar or sodium alone does not prove that a product is healthier."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**{name1}**")
        st.metric(
            "Sugar",
            f"{format_number(sugar1)} g" if sugar1 is not None else "Not listed",
        )
        st.metric(
            "Sodium",
            f"{format_number(sodium1)} mg" if sodium1 is not None else "Not listed",
        )

    with col2:
        st.markdown(f"**{name2}**")
        st.metric(
            "Sugar",
            f"{format_number(sugar2)} g" if sugar2 is not None else "Not listed",
        )
        st.metric(
            "Sodium",
            f"{format_number(sodium2)} mg" if sodium2 is not None else "Not listed",
        )

    advantages1 = []
    advantages2 = []

    if sugar1 is not None and sugar2 is not None:
        if sugar1 < sugar2:
            advantages1.append("lower recorded sugar")
        elif sugar2 < sugar1:
            advantages2.append("lower recorded sugar")

    if sodium1 is not None and sodium2 is not None:
        if sodium1 < sodium2:
            advantages1.append("lower recorded sodium")
        elif sodium2 < sodium1:
            advantages2.append("lower recorded sodium")

    if advantages1 and not advantages2:
        st.success(
            f"**{name1}** compares favourably on the measured factors: "
            + ", ".join(advantages1) + "."
        )
    elif advantages2 and not advantages1:
        st.success(
            f"**{name2}** compares favourably on the measured factors: "
            + ", ".join(advantages2) + "."
        )
    elif advantages1 and advantages2:
        st.info(
            f"**It's a trade-off.** {name1} has "
            + ", ".join(advantages1)
            + ", while "
            + name2
            + " has "
            + ", ".join(advantages2)
            + "."
        )
    else:
        st.info(
            "There isn't enough comparable sugar and sodium data "
            "to choose a winner on these factors."
        )

    st.caption(
        "Values are from the available dataset. Confirm the nutrition basis "
        "on both packages and consider serving size, saturated fat, fibre, "
        "protein, ingredients, and allergens before deciding."
    )


# =========================================================
# GREETING
# =========================================================

def greeting():
    return """
Hey there! 👋 Welcome to **IngredientIQ**.

I can help you:

- 🧾 Show the ingredients listed on a food package
- 🌿 Explain ingredients in simple English
- 📊 Check nutrition values
- ⚗️ Explain additives and INS codes
- 🥜 Check listed allergens
- ⚖️ Compare two products and summarise the differences

You can ask follow-up questions without repeating the product name every time.

**Try:** “What's inside Britannia Little Hearts?”
"""


# =========================================================
# HANDLE USER QUESTIONS
# =========================================================

def handle_question(question):
    # Exclude the current user message from previous conversation history.
    previous_messages = st.session_state.messages[:-1]

    history = [
        {"role": message["role"], "content": message["content"]}
        for message in previous_messages[-8:]
        if message["role"] in {"user", "assistant"}
    ]

    intent = detect_intent(question)

    # -----------------------------------------------------
    # 1. COMPARE TWO PRODUCTS
    # -----------------------------------------------------
    if intent == "compare":
        pair = find_two_products(question)

        if len(pair) != 2:
            return (
                "I couldn't confidently identify both products, so I haven't "
                "substituted other products by mistake. 😊\n\n"
                "Please provide both product names, for example:\n\n"
                "**Compare Parle-G and Britannia Little Hearts**"
            )

        set_active_product(pair[0])

        # Render the existing comparison table.
        comparison = compare_products(pair[0], pair[1])

        if comparison is not None:
            if isinstance(comparison, str):
                st.markdown(comparison)
            else:
                st.write(comparison)

        # Render the summary below the comparison table.
        comparison_summary(pair[0], pair[1])

        return (
            f"Comparison displayed for **{product_name(pair[0])}** "
            f"and **{product_name(pair[1])}**, including the nutrition summary."
        )

    # -----------------------------------------------------
    # 2. RESOLVE ACTIVE PRODUCT
    # -----------------------------------------------------
    explicit_product = find_explicit_product(question)

    if explicit_product is not None:
        set_active_product(explicit_product)
        product = explicit_product
    else:
        product = get_active_product()

    if intent == "greeting" and explicit_product is None:
        return greeting()

    # -----------------------------------------------------
    # 3. INGREDIENT LIST
    # -----------------------------------------------------
    if product is not None and is_ingredient_list_request(question):
        return ingredient_list_only(product)

    # -----------------------------------------------------
    # 4. EXPLAIN INGREDIENTS
    # -----------------------------------------------------
    if product is not None and is_ingredient_explanation_request(question):
        context = explanation_context(product)

        instructions = (
            "Explain the ingredients for the active product in simple, "
            "natural English. Use the supplied CSV Ingredient_Explanation "
            "column as the primary source. Make a Markdown table with "
            "columns Ingredient, What it does (in simple words), and Why it "
            "is used. Include identifiable ingredients from the ingredient "
            "list. Do not invent functions that the data does not support; "
            "say when a function is not specified. Do not add a nutrition "
            "table. Mention listed allergens separately. Begin with one "
            "friendly sentence."
        )

        return answer_with_groq(
            question=f"{instructions}\n\nUser question: {question}",
            context=context,
            intent="ingredients",
            product_name=product_name(product),
            history=history,
        )

    # -----------------------------------------------------
    # 5. PALM OIL
    # -----------------------------------------------------
    if product is not None and is_palm_oil_question(question):
        ingredients = clean_value(product.get("Ingredients"))
        palm_field = clean_value(product.get("Added palm oil"))

        if ingredients != "Not listed" and re.search(
            r"\bpalm(?:\s+oil)?\b|\bpalmolein\b",
            ingredients,
            re.IGNORECASE,
        ):
            return (
                f"**Yes.** The ingredient list for **{product_name(product)}** "
                "explicitly mentions palm oil or palmolein.\n\n"
                f"**Listed ingredients:** {ingredients}"
            )

        if palm_field != "Not listed":
            return (
                f"**Palm oil information for {product_name(product)}:** "
                f"{palm_field}\n\n"
                "This is the database classification. Check the current "
                "package label to confirm the ingredients."
            )

        if ingredients != "Not listed":
            return (
                f"The available ingredient list for **{product_name(product)}** "
                "does not explicitly mention palm oil. This does not guarantee "
                "that the product is palm-oil-free; verify the current label."
            )

        return (
            f"Palm oil information for **{product_name(product)}** "
            "is not available in the database."
        )

    # -----------------------------------------------------
    # 6. ALLERGENS
    # -----------------------------------------------------
    if product is not None and is_allergen_question(question):
        allergens = clean_value(product.get("Allergens"))

        return (
            f"### Allergens listed for {product_name(product)}\n\n"
            f"{allergens}\n\n"
            "Check the current package for updated allergen and "
            "cross-contact warnings."
        )

    # -----------------------------------------------------
    # 7. NUTRITION
    # -----------------------------------------------------
    if product is not None and is_nutrition_question(question):
        return nutrition_answer(question, product)

    # -----------------------------------------------------
    # 8. ASK FOR A PRODUCT IF NEEDED
    # -----------------------------------------------------
    if product is None and intent in {
        "nutrition", "categories", "allergens", "ingredients"
    }:
        return (
            "Which product are you asking about? Tell me its name once, "
            "and I'll use it for your follow-up questions."
        )

    # -----------------------------------------------------
    # 9. RETRIEVER + GROQ FOR OTHER QUESTIONS
    # -----------------------------------------------------
    context = retrieve_context(
        question,
        retriever,
        product=product,
        top_k=6,
    )

    if product is not None:
        context = (
            f"ACTIVE PRODUCT DATA:\n"
            f"Product: {product_name(product)}\n"
            f"Ingredients: {clean_value(product.get('Ingredients'))}\n"
            f"Ingredient_Explanation: "
            f"{clean_value(product.get('Ingredient_Explanation'))}\n"
            f"Allergens: {clean_value(product.get('Allergens'))}\n\n"
            f"RETRIEVED REFERENCE CONTEXT:\n{context}"
        )

    return answer_with_groq(
        question=question,
        context=context,
        intent="additive" if is_additive_question(question) else intent,
        product_name=product_name(product) if product is not None else None,
        history=history,
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">🥣</div>
            <div>
                <div class="brand-name">Ingredient<span>IQ</span></div>
                <div class="brand-tagline">Know what's in your food.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "✦   New Chat",
        type="primary",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.session_state.active_product_name = None
        st.session_state.pending_question = None
        st.rerun()

    st.markdown(
        '<div class="section-heading">💡 Example questions</div>',
        unsafe_allow_html=True,
    )

    examples = [
        ("🔎", "What's inside Britannia Little Hearts?"),
        ("🌿", "Does it contain palm oil?"),
        ("📊", "How much sugar does it have?"),
        ("⚗️", "Explain INS 322"),
        ("🥜", "What allergens are listed?"),
        ("⚖️", "Compare Parle-G and Britannia Little Hearts"),
    ]

    for i, (emoji, example) in enumerate(examples):
        if st.button(
            f"{emoji}  {example}",
            key=f"example_{i}",
            use_container_width=True,
        ):
            st.session_state.pending_question = example

    st.divider()

    active = get_active_product()

    if active is not None:
        st.markdown("**📌 Current product**")
        st.markdown(
            f"<div>{product_name(active)}</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="small-note">
            IngredientIQ uses the available product database.
            Always check the current package label.
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Ingredient<span>IQ</span></div>
        <div class="hero-subtitle">
            A little more knowledge about what's on your plate.
        </div>
        <div class="hero-description">
            Ask about ingredients, nutrition, additives, allergens,
            and product differences.
        </div>
        <div class="pill-row">
            <div class="pill">🌿 Ingredients</div>
            <div class="pill">📊 Nutrition</div>
            <div class="pill">⚗️ Food additives</div>
            <div class="pill">🥜 Allergens</div>
            <div class="pill">⚖️ Compare</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# EMPTY-CHAT WELCOME
# =========================================================

if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-card">
            <h3>🌱 Understand your food better</h3>
            <p>
                Start with a product name, ask about an ingredient,
                check nutrition, or compare two products.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# =========================================================
# CHAT INPUT AND RESPONSE
# =========================================================

question = st.chat_input("Ask about your food product...")

if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None


if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Understanding your food label..."):
            try:
                response = handle_question(question)
            except Exception:
                import traceback
                traceback.print_exc()

                response = (
                    "Sorry, something went wrong. Please check the "
                    "VS Code terminal for the technical error."
                )

            st.markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
    })
