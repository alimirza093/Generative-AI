import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional


load_dotenv()

model = ChatMistralAI(
    model="open-mistral-7b",
    temperature=0.7
)


class ProductData(BaseModel):
    name: str
    brand: str
    category: str
    price: Optional[float]
    currency: Optional[str]
    rating: Optional[float]
    features: List[str]
    description: str


parser = PydanticOutputParser(pydantic_object=ProductData)

prompt = ChatPromptTemplate([
    ("system", """
        Extract useful product information from the paragraph.
        {format_instructions}
    """),
    ("human", "{paragraph}")
])


def extract_product_data(paragraph_text: str):

    final_prompt = prompt.invoke({
        "format_instructions": parser.get_format_instructions(),
        "paragraph": paragraph_text
    })

    response = model.invoke(final_prompt)

    product_data = parser.parse(response.content)

    return (
        product_data.model_dump()
        if hasattr(product_data, "model_dump")
        else product_data.dict()
    )


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Product Data Extractor",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# -----------------------------
# Custom CSS
# -----------------------------

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: #0b0f14;
        color: #f5f7fa;
    }

    /* Main container */
    .block-container {
        max-width: 1200px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .hero {
        text-align: center;
        padding: 1rem 0 2.5rem 0;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0.5rem;
    }

    .hero-subtitle {
        color: #9ca3af;
        font-size: 1.05rem;
    }

    /* Input label */
    .input-label {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }

    /* Text area */
    textarea {
        background-color: #111827 !important;
        color: #f9fafb !important;
        border: 1px solid #263241 !important;
        border-radius: 12px !important;
    }

    textarea:focus {
        border: 1px solid #6366f1 !important;
        box-shadow: 0 0 0 1px #6366f1 !important;
    }

    /* Extract button */
    .stButton > button {
        width: 100%;
        height: 3rem;
        border-radius: 10px;
        border: none;
        font-size: 1rem;
        font-weight: 700;
        background: #6366f1;
        color: white;
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        background: #4f46e5;
        transform: translateY(-1px);
    }

    /* Section title */
    .section-title {
        font-size: 1.5rem;
        font-weight: 750;
        margin-top: 2.5rem;
        margin-bottom: 1.2rem;
    }

    /* Info cards */
    .card {
        background: #111827;
        border: 1px solid #263241;
        border-radius: 14px;
        padding: 1.2rem;
        min-height: 110px;
    }

    .card-label {
        color: #9ca3af;
        font-size: 0.85rem;
        margin-bottom: 0.4rem;
    }

    .card-value {
        color: #f9fafb;
        font-size: 1.15rem;
        font-weight: 650;
    }

    /* Large product title */
    .product-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }

    .product-meta {
        color: #9ca3af;
        font-size: 0.95rem;
    }

    /* Description */
    .summary {
        background: #111827;
        border-left: 4px solid #6366f1;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        line-height: 1.7;
        color: #d1d5db;
    }

    /* Chips */
    .chip {
        display: inline-block;
        background: #1f2937;
        border: 1px solid #374151;
        color: #dbeafe;
        border-radius: 999px;
        padding: 0.35rem 0.75rem;
        margin: 0.2rem 0.25rem 0.2rem 0;
        font-size: 0.85rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280;
        margin-top: 4rem;
        font-size: 0.85rem;
    }

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Hero Section
# -----------------------------

st.title("🛍️ Product Data Extractor")
st.caption("Extract structured product information from unstructured text using AI")


# -----------------------------
# Input Section
# -----------------------------

st.subheader("Product Paragraph")

paragraph = st.text_area(
    label="Product Paragraph",
    label_visibility="collapsed",
    placeholder=(
        "Paste a product paragraph here...\n\n"
        "Example: The Dell XPS 15 is a premium laptop..."
    ),
    height=240
)

st.write("")

extract = st.button("🎯 Extract Product Information")


# -----------------------------
# Extraction
# -----------------------------

if extract:

    if not paragraph.strip():

        st.warning("Please enter a product paragraph first.")

    else:

        with st.spinner("Analyzing product information..."):

            try:
                product_data = extract_product_data(paragraph)

                st.success(
                    "Product information extracted successfully."
                )

            except Exception as exc:

                st.warning(
                    f"Model extraction failed. Showing a fallback summary. Error: {exc}"
                )

                product_data = {
                    "name": "Product Information",
                    "brand": "Not provided",
                    "category": "Not provided",
                    "price": None,
                    "currency": None,
                    "rating": None,
                    "features": ["Not provided"],
                    "description": paragraph.strip()
                }


        # -----------------------------
        # Results
        # -----------------------------

        st.subheader("Extracted Information")

        st.markdown(f"## {product_data['name']}")

        st.caption(
            f"{product_data['brand']} • {product_data['category']}"
        )

        st.write("")


        # -----------------------------
        # Basic Information Cards
        # -----------------------------

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"**Brand**\n\n{product_data['brand']}"
            )

        with col2:

            price = product_data["price"]
            currency = product_data["currency"]

            if price is not None:

                price_text = (
                    f"{currency} {price}"
                    if currency
                    else str(price)
                )

            else:
                price_text = "Not available"

            st.markdown(
                f"**Price**\n\n{price_text}"
            )

        with col3:

            rating = product_data["rating"]

            rating_text = (
                f"⭐ {rating}/5"
                if rating is not None
                else "Not available"
            )

            st.markdown(
                f"**Rating**\n\n{rating_text}"
            )


        # -----------------------------
        # Category
        # -----------------------------

        st.subheader("Category")

        st.markdown(product_data["category"])


        # -----------------------------
        # Features
        # -----------------------------

        st.subheader("Features")

        st.markdown(
            " • ".join(product_data["features"])
        )


        # -----------------------------
        # Description
        # -----------------------------

        st.subheader("Description")

        st.markdown(
            f"> {product_data['description']}"
        )


# -----------------------------
# Footer
# -----------------------------

st.caption(
    "AI-powered structured data extraction • "
    "LangChain + Mistral + Pydantic"
)
import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional


load_dotenv()

model = ChatMistralAI(
    model="open-mistral-7b",
    temperature=0.7
)


class ProductData(BaseModel):
    name: str
    brand: str
    category: str
    price: Optional[float]
    currency: Optional[str]
    rating: Optional[float]
    features: List[str]
    description: str


parser = PydanticOutputParser(pydantic_object=ProductData)

prompt = ChatPromptTemplate([
    ("system", """
        Extract useful product information from the paragraph.
        {format_instructions}
    """),
    ("human", "{paragraph}")
])


def extract_product_data(paragraph_text: str):

    final_prompt = prompt.invoke({
        "format_instructions": parser.get_format_instructions(),
        "paragraph": paragraph_text
    })

    response = model.invoke(final_prompt)

    product_data = parser.parse(response.content)

    return (
        product_data.model_dump()
        if hasattr(product_data, "model_dump")
        else product_data.dict()
    )


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Product Data Extractor",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# -----------------------------
# Custom CSS
# -----------------------------

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: #0b0f14;
        color: #f5f7fa;
    }

    /* Main container */
    .block-container {
        max-width: 1200px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .hero {
        text-align: center;
        padding: 1rem 0 2.5rem 0;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -1px;
        margin-bottom: 0.5rem;
    }

    .hero-subtitle {
        color: #9ca3af;
        font-size: 1.05rem;
    }

    /* Input label */
    .input-label {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }

    /* Text area */
    textarea {
        background-color: #111827 !important;
        color: #f9fafb !important;
        border: 1px solid #263241 !important;
        border-radius: 12px !important;
    }

    textarea:focus {
        border: 1px solid #6366f1 !important;
        box-shadow: 0 0 0 1px #6366f1 !important;
    }

    /* Extract button */
    .stButton > button {
        width: 100%;
        height: 3rem;
        border-radius: 10px;
        border: none;
        font-size: 1rem;
        font-weight: 700;
        background: #6366f1;
        color: white;
        transition: 0.2s ease;
    }

    .stButton > button:hover {
        background: #4f46e5;
        transform: translateY(-1px);
    }

    /* Section title */
    .section-title {
        font-size: 1.5rem;
        font-weight: 750;
        margin-top: 2.5rem;
        margin-bottom: 1.2rem;
    }

    /* Info cards */
    .card {
        background: #111827;
        border: 1px solid #263241;
        border-radius: 14px;
        padding: 1.2rem;
        min-height: 110px;
    }

    .card-label {
        color: #9ca3af;
        font-size: 0.85rem;
        margin-bottom: 0.4rem;
    }

    .card-value {
        color: #f9fafb;
        font-size: 1.15rem;
        font-weight: 650;
    }

    /* Large product title */
    .product-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }

    .product-meta {
        color: #9ca3af;
        font-size: 0.95rem;
    }

    /* Description */
    .summary {
        background: #111827;
        border-left: 4px solid #6366f1;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        line-height: 1.7;
        color: #d1d5db;
    }

    /* Chips */
    .chip {
        display: inline-block;
        background: #1f2937;
        border: 1px solid #374151;
        color: #dbeafe;
        border-radius: 999px;
        padding: 0.35rem 0.75rem;
        margin: 0.2rem 0.25rem 0.2rem 0;
        font-size: 0.85rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280;
        margin-top: 4rem;
        font-size: 0.85rem;
    }

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Hero Section
# -----------------------------

st.title("🛍️ Product Data Extractor")
st.caption("Extract structured product information from unstructured text using AI")


# -----------------------------
# Input Section
# -----------------------------

st.subheader("Product Paragraph")

paragraph = st.text_area(
    label="Product Paragraph",
    label_visibility="collapsed",
    placeholder=(
        "Paste a product paragraph here...\n\n"
        "Example: The Dell XPS 15 is a premium laptop..."
    ),
    height=240
)

st.write("")

extract = st.button("🎯 Extract Product Information")


# -----------------------------
# Extraction
# -----------------------------

if extract:

    if not paragraph.strip():

        st.warning("Please enter a product paragraph first.")

    else:

        with st.spinner("Analyzing product information..."):

            try:
                product_data = extract_product_data(paragraph)

                st.success(
                    "Product information extracted successfully."
                )

            except Exception as exc:

                st.warning(
                    f"Model extraction failed. Showing a fallback summary. Error: {exc}"
                )

                product_data = {
                    "name": "Product Information",
                    "brand": "Not provided",
                    "category": "Not provided",
                    "price": None,
                    "currency": None,
                    "rating": None,
                    "features": ["Not provided"],
                    "description": paragraph.strip()
                }


        # -----------------------------
        # Results
        # -----------------------------

        st.subheader("Extracted Information")

        st.markdown(f"## {product_data['name']}")

        st.caption(
            f"{product_data['brand']} • {product_data['category']}"
        )

        st.write("")


        # -----------------------------
        # Basic Information Cards
        # -----------------------------

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"**Brand**\n\n{product_data['brand']}"
            )

        with col2:

            price = product_data["price"]
            currency = product_data["currency"]

            if price is not None:

                price_text = (
                    f"{currency} {price}"
                    if currency
                    else str(price)
                )

            else:
                price_text = "Not available"

            st.markdown(
                f"**Price**\n\n{price_text}"
            )

        with col3:

            rating = product_data["rating"]

            rating_text = (
                f"⭐ {rating}/5"
                if rating is not None
                else "Not available"
            )

            st.markdown(
                f"**Rating**\n\n{rating_text}"
            )


        # -----------------------------
        # Category
        # -----------------------------

        st.subheader("Category")

        st.markdown(product_data["category"])


        # -----------------------------
        # Features
        # -----------------------------

        st.subheader("Features")

        st.markdown(
            " • ".join(product_data["features"])
        )


        # -----------------------------
        # Description
        # -----------------------------

        st.subheader("Description")

        st.markdown(
            f"> {product_data['description']}"
        )


# -----------------------------
# Footer
# -----------------------------

st.caption(
    "AI-powered structured data extraction • "
    "LangChain + Mistral + Pydantic"
)