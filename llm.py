
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

# Load .env from the folder containing this file
PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env", override=True)


def answer_with_groq(
    question,
    context,
    intent="general",
    product_name=None,
    history=None
):
    api_key = os.getenv("GROQ_API_KEY", "").strip()

    if not api_key or api_key == "your_actual_groq_api_key":
        return (
            "I can't find a valid Groq API key. "
            "Please check the .env file in your project folder."
        )

    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    client = Groq(api_key=api_key)

    system_prompt = """
You are IngredientIQ, a friendly, professional food-label
assistant. Explain food information in simple, natural English.

PERSONALITY
- Be warm, clear, polite, and conversational.
- Answer the exact question first.
- Keep simple answers to 1-3 sentences.
- Explain technical terms in everyday language.
- Avoid filler, repetitive warnings, and database language.

ANSWER FORMAT
- Specific question: answer only what was asked.
- One ingredient: explain only that ingredient.
- Multiple ingredients: use a clear table with columns
  "Ingredient" and "What it does".
- Nutrition: provide only the requested nutrient and its
  recorded serving basis. Show a full table only when asked.
- Additives: explain what the additive is and why it is used.
- Allergens: list recorded allergens and advise checking
  the current package label.
- Comparisons: use a compact table and compare only the
  two products explicitly requested.

PRODUCT CONTEXT
- Use conversation history to understand follow-up questions.
- "It", "that", and similar words refer to the current product
  unless the user names a different product.
- Switch to a newly named product when appropriate.
- Never substitute another product in a comparison.
- If either comparison product is unclear, ask for clarification.

DATA ACCURACY
- Use supplied context for product-specific facts.
- Never invent ingredients, nutrition values, allergens,
  additive presence, or regulatory limits.
- "Not listed" does not prove absence.
- If information is missing, say so briefly and naturally.
- Preserve recorded values and units.
- Do not claim that a product is safe for a specific allergy.

STYLE
- Do not show raw database fields or retrieval details.
- Avoid tables for simple yes/no questions.
- Use tables when explaining multiple ingredients or comparing
  products, not automatically for every question.
- Ask a clarification only when genuinely necessary.
- Do not force a follow-up question after every answer.

Your priority is to answer the actual question accurately,
naturally, and in the right amount of detail.
"""

    messages = [
        {"role": "system", "content": system_prompt}
    ]

    # Include recent conversation for follow-up understanding
    if history:
        messages.extend(history[-8:])

    user_message = f"""
Active product: {product_name or "No product identified"}

Question:
{question}

Intent:
{intent}

Retrieved product information:
{str(context)[:18000]}

Answer only the question asked. Use the supplied information
for product-specific facts. If it is unavailable, say so.
"""

    messages.append({
        "role": "user",
        "content": user_message
    })

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=messages
        )

        return response.choices[0].message.content.strip()

    except Exception as error:
        print(f"Groq API error: {error}")
        return (
            "Sorry, I couldn't generate an answer just now. "
            "Please check the terminal for the technical error."
        )
