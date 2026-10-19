
import re
from difflib import SequenceMatcher


def normalize(text):
    """Normalize product names and questions for matching."""
    text = str(text or "").lower().replace("\u00a0", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def product_names(row):
    """Get the product's known names and brand."""
    names = []

    for column in ("Item name", "Brand_Name"):
        value = row.get(column, "")
        if value is not None:
            value = str(value).strip()
            if value and value.lower() not in {"nan", "none"}:
                names.append(value)

    return list(dict.fromkeys(names))


def _tokens(text):
    return set(normalize(text).split())


def _name_score(query, name):
    """Score a query against one product name."""
    q = normalize(query)
    n = normalize(name)

    if not q or not n:
        return 0.0

    if q == n:
        return 1.0

    # Match a complete name anywhere in a longer question.
    if re.search(rf"(?<!\w){re.escape(n)}(?!\w)", q):
        return 1.0

    q_tokens = set(q.split())
    n_tokens = set(n.split())

    # Require meaningful token overlap, not just generic words.
    meaningful = {
        "parle", "britannia", "good", "day", "little", "hearts",
        "cashew", "cookies", "cookie", "biscuits", "biscuit",
        "gluco", "original"
    }

    overlap = q_tokens & n_tokens
    if not overlap:
        return 0.0

    coverage = len(overlap) / max(len(n_tokens), 1)
    precision = len(overlap) / max(len(q_tokens), 1)
    similarity = SequenceMatcher(None, q, n).ratio()

    # Short queries such as "biscuits" are too vague to identify a product.
    if len(overlap) == 1:
        token = next(iter(overlap))
        if token not in meaningful:
            return 0.0

    return max(
        coverage * 0.88,
        (coverage * 0.65 + precision * 0.20 + similarity * 0.15)
    )


def find_product(query, products, threshold=0.58):
    """
    Return (row, score).

    Exact full product names take priority. If several products have similar
    scores, avoid selecting an arbitrary product.
    """
    q = normalize(query)
    if not q or products is None or products.empty:
        return None, 0.0

    candidates = []

    for index, row in products.iterrows():
        best_score = max(
            (_name_score(q, name) for name in product_names(row)),
            default=0.0
        )
        candidates.append((best_score, index, row))

    candidates.sort(key=lambda item: item[0], reverse=True)

    best_score, _, best_row = candidates[0]

    if best_score < threshold:
        return None, best_score

    # If two different products have almost identical scores, don't guess.
    if len(candidates) > 1:
        second_score, _, second_row = candidates[1]
        if (
            second_score >= threshold
            and best_score - second_score < 0.06
            and normalize(best_row.get("Item name", ""))
            != normalize(second_row.get("Item name", ""))
        ):
            return None, best_score

    return best_row, best_score


def find_two_products(query, products):
    """
    Find two explicitly mentioned products in the question.
    Uses exact names first, then checks distinctive product-name phrases.
    """
    q = normalize(query)
    if not q or products is None or products.empty:
        return []

    found = []

    for index, row in products.iterrows():
        names = product_names(row)
        matched = []

        for name in names:
            n = normalize(name)
            if not n:
                continue

            # Full product name explicitly appears in the question.
            if re.search(rf"(?<!\w){re.escape(n)}(?!\w)", q):
                matched.append((len(n), 1.0, n))
                continue

            # Permit shortened names such as "Little Hearts".
            # Require at least two distinctive tokens for a shortened name.
            tokens = n.split()
            distinctive = [
                token for token in tokens
                if token not in {
                    "the", "original", "classic", "family", "pack",
                    "and", "with", "of", "a", "an"
                }
            ]

            if len(distinctive) >= 2:
                phrase = " ".join(distinctive)
                if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", q):
                    matched.append((len(phrase), 0.9, phrase))
                else:
                    present = sum(
                        token in q.split() for token in distinctive
                    )
                    if (
                        present >= 2
                        and present / len(distinctive) >= 0.67
                    ):
                        matched.append((present, 0.7, phrase))

        if matched:
            matched.sort(reverse=True)
            found.append((matched[0][0], matched[0][1], index, row))

    # Prefer stronger and longer matches, while preserving question order
    # for exact full-name matches where possible.
    found.sort(key=lambda item: (item[1], item[0]), reverse=True)

    result = []
    seen = set()

    for _, _, index, row in found:
        if index not in seen:
            result.append(row)
            seen.add(index)

        if len(result) == 2:
            break

    return result


def detect_intent(query):
    """Classify the question, prioritizing specific additive-code queries."""
    q = normalize(query)

    if not q:
        return "general"

    if q in {"hi", "hello", "hey", "hii", "heyy"} or any(
        phrase in q for phrase in (
            "good morning", "good afternoon", "good evening",
            "how are you", "who are you", "what can you do",
            "help me get started"
        )
    ):
        return "greeting"

    if any(term in q for term in (
        "compare", "comparison", "versus", " vs ",
        "difference between", "which is better"
    )):
        return "compare"

    # INS codes and explicit additive questions must be checked before
    # broad words such as "emulsifier", "sodium", or "nutrition".
    if (
        re.search(r"\bins?\s*\-?\s*\d+\s*[a-z]?\b", q)
        or "fssai" in q
        or "ins code" in q
        or "what does ins" in q
        or "what is ins" in q
        or "additive sodium" in q
        or "sodium additive" in q
        or "which additive" in q
        or "what additive" in q
    ):
        return "additive"

    if any(term in q for term in (
        "allergen", "allergens", "allergy", "allergies",
        "gluten", "milk allergy", "contains nuts"
    )):
        return "allergens"

    if any(term in q for term in (
        "palm oil", "added sugar", "preservative", "preservatives",
        "colour", "color", "stabilizer", "stabilizers",
        "emulsifier", "emulsifiers", "thickener", "thickeners",
        "sweetener", "sweeteners", "raising agent", "raising agents",
        "additive categories"
    )):
        return "categories"

    if any(term in q for term in (
        "calorie", "calories", "protein", "sugar content",
        "sodium", "fat content", "nutrition", "nutritional",
        "nutrient", "nutrition facts", "how much sugar",
        "how much fat", "how many calories", "how much protein",
        "how much sodium"
    )):
        return "nutrition"

    if any(term in q for term in (
        "ingredient", "ingredients", "made of", "made from",
        "what is in", "what s in", "composition",
        "break down", "explain them", "explain those",
        "explain these", "what goes into"
    )):
        return "ingredients"

    return "general"
