import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional

load_dotenv()

model = ChatMistralAI(model="open-mistral-7b", temperature=0.7)


class MovieData(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: str
    cast: List[str]
    rating: Optional[float]
    summary: str


parser = PydanticOutputParser(pydantic_object=MovieData)

prompt = ChatPromptTemplate([
    ("system", """
        Extract useful movie information from the paragraph.
        {format_instructions}
    """),
    ("human", "{paragraph}")
])


def extract_movie_data(paragraph_text: str):
    final_prompt = prompt.invoke({
        "format_instructions": parser.get_format_instructions(),
        "paragraph": paragraph_text
    })

    response = model.invoke(final_prompt)
    movie_data = parser.parse(response.content)
    return movie_data.model_dump() if hasattr(movie_data, "model_dump") else movie_data.dict()


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Movie Data Extractor",
    page_icon="🎬",
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

    /* Large movie title */
    .movie-title {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }

    .movie-meta {
        color: #9ca3af;
        font-size: 0.95rem;
    }

    /* Summary */
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

st.title("🎬 Movie Data Extractor")
st.caption("Extract structured movie information from unstructured text using AI")


# -----------------------------
# Input Section
# -----------------------------

st.subheader("Movie Paragraph")

paragraph = st.text_area(
    label="Movie Paragraph",
    label_visibility="collapsed",
    placeholder=(
        "Paste a movie paragraph here...\n\n"
        "Example: The Dark Knight is a 2008 superhero film..."
    ),
    height=240
)

st.write("")

extract = st.button("🎯 Extract Movie Information")


# -----------------------------
# Extraction
# -----------------------------

if extract:

    if not paragraph.strip():

        st.warning("Please enter a movie paragraph first.")

    else:

        with st.spinner("Analyzing movie information..."):
            try:
                movie_data = extract_movie_data(paragraph)
                st.success("Movie information extracted successfully.")
            except Exception as exc:
                st.warning(f"Model extraction failed. Showing a fallback summary. Error: {exc}")
                movie_data = {
                    "title": "Movie Information",
                    "release_year": None,
                    "genre": ["Unspecified"],
                    "director": "Not provided",
                    "cast": ["Not provided"],
                    "rating": None,
                    "summary": paragraph.strip()
                }


        # -----------------------------
        # Results
        # -----------------------------

        st.subheader("Extracted Information")
        st.markdown(f"## {movie_data['title']}")
        st.caption(f"{movie_data['release_year']}")

        st.write("")


        # -----------------------------
        # Basic Information Cards
        # -----------------------------

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"**Director**\n\n{movie_data['director']}")

        with col2:
            rating = movie_data["rating"]
            rating_text = (
                f"⭐ {rating}/10"
                if rating is not None
                else "Not available"
            )
            st.markdown(f"**Rating**\n\n{rating_text}")

        with col3:
            st.markdown(f"**Release Year**\n\n{movie_data['release_year']}")


        # -----------------------------
        # Genre
        # -----------------------------

        st.subheader("Genres")
        st.markdown(" • ".join(movie_data["genre"]))


        # -----------------------------
        # Cast
        # -----------------------------

        st.subheader("Cast")
        st.markdown(" • ".join(movie_data["cast"]))


        # -----------------------------
        # Summary
        # -----------------------------

        st.subheader("Summary")
        st.markdown(f"> {movie_data['summary']}")


# -----------------------------
# Footer
# -----------------------------

st.caption("AI-powered structured data extraction • LangChain + Mistral + Pydantic")