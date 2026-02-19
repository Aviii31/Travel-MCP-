from __future__ import annotations

import os
from datetime import date, timedelta
from typing import List

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    import google.genai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from travel_tools_server import get_weather_impl, search_travel_options_impl



CITY_INTRO_TEMPLATE = """
You are a helpful, detail-oriented travel planning assistant with expertise in visual storytelling.

Create a concise trip plan using the structured data below. Include vivid descriptions of attractions and experiences suitable for photography and visual discovery.

User request:
{user_request}

Destination city: {city}
Origin city: {origin}
Trip length (days): {num_days}
Travel dates: {travel_dates}

Weather information (current + during trip):
{weather_summary}

Flight & hotel options (these may be mock estimates but are still useful for planning):
{travel_options}

Trip preferences or constraints (if any):
{preferences}

Your task:
1. Start with a single paragraph about the city's cultural and historic significance with visual highlights.
2. Summarize the current weather and expected conditions over the trip dates.
3. Clearly list the travel dates and highlight 2–3 good flight and hotel options using the given data.
4. Propose a practical, engaging day-by-day itinerary for the full trip duration.
5. Include specific, photogenic landmarks, attractions, and experiences with descriptive details that bring them to life.

Format the answer in markdown with the following sections and headings:

## City Overview
<one paragraph with vivid descriptions and key visual attractions>

## Weather Summary
<short bullet list summarizing current weather + forecast over the trip>

## Travel Logistics (Flights & Hotels)
- **Travel dates**: ...
- **Flights**: ...
- **Hotels**: ...

## Day-by-Day Plan
- **Day 1**: [Morning/Afternoon/Evening activities with specific landmark names and descriptions]
- **Day 2**: [Include photo-worthy locations and experiences]
- etc.

Remember to include specific attraction names, neighborhood descriptions, and memorable experiences that travelers will want to photograph and remember.
"""



def _build_default_user_request(city: str, num_days: int, month_label: str | None) -> str:
    if month_label:
        return f"Plan a {num_days}-day trip to {city} in {month_label}."
    return f"Plan a {num_days}-day trip to {city}."


def _compute_date_range(start: date, num_days: int) -> List[date]:
    return [start + timedelta(days=i) for i in range(num_days)]


def _get_weather_block(city: str, trip_dates: List[date]) -> str:
    lines: list[str] = []

    # Current weather (today)
    today_str = date.today().isoformat()
    lines.append(f"Current weather (today, {today_str}):")
    lines.append(get_weather_impl(city, today_str))
    lines.append("")

    # Weather during the trip
    lines.append("Weather during trip:")
    for d in trip_dates:
        lines.append(get_weather_impl(city, d.isoformat()))
    return "\n".join(lines)




def _get_travel_options(origin: str, destination: str, depart: date, return_date: date) -> str:
    return search_travel_options_impl(
        origin=origin,
        destination=destination,
        depart_date=depart.isoformat(),
        return_date=return_date.isoformat(),
    )


def _get_llm(model_name: str = "gemini-2.5-flash") -> ChatGoogleGenerativeAI:
    """
    Returns a Gemini chat model via LangChain.

    For this demo we use Gemini via GOOGLE_API_KEY.
    """
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY environment variable is not set. "
            "Set it to a valid Gemini API key to run the app."
        )
    # Common model names to try:
    # - gemini-1.5-flash (latest)
    # - gemini-1.5-pro (latest)
    # - gemini-pro (older, more widely available)
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.4,
        max_output_tokens=2048,
    )


def run_planner(
    city: str,
    origin: str,
    start_date: date,
    num_days: int,
    month_label: str | None,
    preferences: str,
    model_name: str = "gemini-2.5-flash",
) -> str:
    llm = _get_llm(model_name=model_name)

    trip_dates = _compute_date_range(start_date, num_days)
    travel_dates_str = f"{trip_dates[0].isoformat()} to {trip_dates[-1].isoformat()}"

    weather_summary = _get_weather_block(city, trip_dates)
    travel_options = _get_travel_options(origin, city, trip_dates[0], trip_dates[-1])

    user_request = _build_default_user_request(city, num_days, month_label)

    prompt = ChatPromptTemplate.from_template(CITY_INTRO_TEMPLATE)
    chain = prompt | llm

    result = chain.invoke(
        {
            "user_request": user_request,
            "city": city,
            "origin": origin,
            "num_days": num_days,
            "travel_dates": travel_dates_str,
            "weather_summary": weather_summary,
            "travel_options": travel_options,
            "preferences": preferences or "No additional preferences provided.",
        }
    )

    # LangChain chat models return a BaseMessage; we want the content string.
    return getattr(result, "content", str(result))


def main() -> None:
    st.set_page_config(
        page_title="AI Trip Planner",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Enhanced Custom CSS
    st.markdown("""
        <style>
        /* Main app background */
        .stApp {
            background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        }
        
        /* Main content area */
        .main .block-container {
            background: transparent;
            padding-top: 2rem;
            max-width: 1200px;
        }
        
        /* Header styling */
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 3rem 2.5rem;
            border-radius: 20px;
            color: white;
            margin-bottom: 2.5rem;
            box-shadow: 0 15px 40px rgba(102, 126, 234, 0.3), 0 0 80px rgba(102, 126, 234, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .main-header h1 {
            color: white !important;
            margin: 0;
            font-size: 3rem;
            font-weight: 800;
            letter-spacing: -1px;
        }
        .main-header p {
            color: rgba(255, 255, 255, 0.95) !important;
            margin-top: 0.75rem;
            font-size: 1.15rem;
            font-weight: 400;
            letter-spacing: 0.3px;
        }
        
        /* Image gallery styling */
        .image-gallery {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }
        
        .image-card {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .image-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        }
        
        .image-card img {
            width: 100%;
            height: 200px;
            object-fit: cover;
            display: block;
        }
        
        .image-caption {
            background: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 0.75rem;
            font-size: 0.85rem;
            text-align: center;
        }
        
        /* Attraction cards */
        .attraction-card {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(102, 126, 234, 0.3);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            transition: all 0.3s ease;
        }
        
        .attraction-card:hover {
            background: rgba(255, 255, 255, 0.12);
            border-color: rgba(102, 126, 234, 0.6);
            transform: translateX(5px);
        }
        
        .attraction-card h4 {
            color: #9d8df1 !important;
            margin-top: 0;
        }
        
        .attraction-images {
            display: flex;
            gap: 0.75rem;
            margin-top: 1rem;
            overflow-x: auto;
        }
        
        .attraction-images img {
            height: 120px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        }
        
        /* Trip plan container */
        .trip-plan-container {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.98), rgba(255, 255, 255, 0.95));
            padding: 3rem;
            border-radius: 18px;
            border-left: 6px solid #667eea;
            margin-top: 2rem;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.25), 0 0 1px rgba(102, 126, 234, 0.2);
            color: #1e1e1e;
            backdrop-filter: blur(10px);
        }
        .trip-plan-container h2,
        .trip-plan-container h3 {
            color: #667eea !important;
            font-weight: 700;
            letter-spacing: -0.5px;
            margin-top: 1.75rem;
            margin-bottom: 1rem;
        }
        .trip-plan-container h2:first-child {
            margin-top: 0;
        }
        .trip-plan-container h4 {
            color: #764ba2 !important;
            font-weight: 600;
        }
        .trip-plan-container p,
        .trip-plan-container li {
            color: #333333 !important;
            line-height: 1.9;
        }
        .trip-plan-container strong {
            color: #667eea !important;
            font-weight: 600;
        }
        
        /* Day card styling */
        .day-card {
            background: rgba(102, 126, 234, 0.08);
            border-left: 4px solid #667eea;
            padding: 1.25rem;
            margin: 1rem 0;
            border-radius: 8px;
        }
        
        .day-card h4 {
            color: #667eea !important;
            margin-top: 0;
        }
        
        /* Sidebar styling */
        .stSidebar {
            background: rgba(15, 12, 41, 0.95);
            border-right: 2px solid rgba(102, 126, 234, 0.2);
        }
        
        /* Sidebar text */
        .stSidebar .stMarkdown,
        .stSidebar h3,
        .stSidebar h2 {
            color: #e0e0e0 !important;
        }
        
        .stSidebar h3 {
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            font-weight: 700;
            letter-spacing: 0.5px;
            font-size: 1.1rem;
        }
        
        .stSidebar hr {
            border-color: rgba(102, 126, 234, 0.25);
            margin: 1.5rem 0;
        }
        
        /* Input fields */
        .stTextInput>div>div>input,
        .stTextArea>div>div>textarea,
        .stSelectbox>div>div>select,
        .stNumberInput>div>div>input {
            background-color: rgba(255, 255, 255, 0.08) !important;
            color: #ffffff !important;
            border: 2px solid rgba(102, 126, 234, 0.35) !important;
            border-radius: 10px;
            transition: all 0.2s ease !important;
            font-size: 0.95rem;
            padding: 0.75rem !important;
        }
        .stTextInput>div>div>input:focus,
        .stTextArea>div>div>textarea:focus,
        .stSelectbox>div>div>select:focus,
        .stNumberInput>div>div>input:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15), inset 0 0 0 1px rgba(102, 126, 234, 0.3) !important;
            background-color: rgba(255, 255, 255, 0.12) !important;
        }
        
        /* Date input */
        .stDateInput>div>div>input {
            background-color: rgba(255, 255, 255, 0.08) !important;
            color: #ffffff !important;
            border: 2px solid rgba(102, 126, 234, 0.35) !important;
            border-radius: 10px;
            transition: all 0.2s ease !important;
        }
        
        /* Selectbox dropdown */
        .stSelectbox>div>div>select {
            background-color: rgba(255, 255, 255, 0.08) !important;
            color: #ffffff !important;
            border: 2px solid rgba(102, 126, 234, 0.35) !important;
            border-radius: 10px;
            transition: all 0.2s ease !important;
        }
        .stSelectbox>div>div>select option {
            background-color: #2a2a3e !important;
            color: #ffffff !important;
        }
        
        /* Labels */
        .stTextInput label,
        .stTextArea label,
        .stSelectbox label,
        .stNumberInput label,
        .stDateInput label {
            color: #e0e0e0 !important;
            font-weight: 600;
            font-size: 0.95rem;
            letter-spacing: 0.3px;
        }
        
        /* Main content text */
        .main .stMarkdown {
            color: #e0e0e0 !important;
        }
        
        /* Headers in main content */
        .main h2 {
            color: #667eea !important;
            border-bottom: 2px solid rgba(102, 126, 234, 0.4);
            padding-bottom: 0.75rem;
            margin-top: 1.75rem;
            margin-bottom: 1rem;
        }
        .main h3 {
            color: #9d8df1 !important;
            margin-top: 1.5rem;
        }
        
        /* Markdown content */
        .stMarkdown {
            line-height: 1.85;
        }
        .stMarkdown ul, .stMarkdown ol {
            margin-left: 1.75rem;
        }
        .stMarkdown li {
            margin-bottom: 0.65rem;
        }
        
        /* Button styling */
        .stButton>button {
            width: 100%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            font-weight: 700;
            padding: 0.95rem !important;
            border-radius: 12px;
            border: none !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            font-size: 1.05rem;
            letter-spacing: 0.5px;
            text-transform: none;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        .stButton>button:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 30px rgba(102, 126, 234, 0.5) !important;
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
        }
        .stButton>button:active {
            transform: translateY(-2px);
        }
        .stButton>button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            background: rgba(102, 126, 234, 0.3) !important;
            transform: none;
            box-shadow: none;
        }
        
        /* Main content text */
        .main .stMarkdown {
            color: #e0e0e0 !important;
        }
        
        /* Headers in main content */
        .main h2 {
            color: #667eea !important;
            border-bottom: 2px solid #667eea;
            padding-bottom: 0.75rem;
            margin-top: 1.5rem;
        }
        .main h3 {
            color: #9d8df1 !important;
            margin-top: 1.25rem;
        }
        
        /* Markdown content */
        .stMarkdown {
            line-height: 1.9;
        }
        .stMarkdown ul, .stMarkdown ol {
            margin-left: 1.75rem;
        }
        .stMarkdown li {
            margin-bottom: 0.6rem;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            color: #e0e0e0 !important;
            background: rgba(102, 126, 234, 0.12);
            border-radius: 10px;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        
        .streamlit-expanderHeader:hover {
            background: rgba(102, 126, 234, 0.18);
        }
        
        /* Error/Success messages */
        .stError {
            background-color: rgba(220, 53, 69, 0.1);
            border-left: 4px solid #dc3545;
            padding: 1.25rem;
            border-radius: 10px;
            border: 1px solid rgba(220, 53, 69, 0.2);
        }
        .stSuccess {
            background-color: rgba(40, 167, 69, 0.1);
            border-left: 4px solid #28a745;
            padding: 1.25rem;
            border-radius: 10px;
            border: 1px solid rgba(40, 167, 69, 0.2);
        }
        .stWarning {
            background-color: rgba(255, 193, 7, 0.1);
            border-left: 4px solid #ffc107;
            padding: 1.25rem;
            border-radius: 10px;
            border: 1px solid rgba(255, 193, 7, 0.2);
        }
        .stInfo {
            background-color: rgba(23, 162, 184, 0.1);
            border-left: 4px solid #17a2b8;
            padding: 1.25rem;
            border-radius: 10px;
            border: 1px solid rgba(23, 162, 184, 0.2);
        }
        
        /* Welcome section */
        .welcome-section {
            background: rgba(255, 255, 255, 0.06);
            border: 2px solid rgba(102, 126, 234, 0.25);
            backdrop-filter: blur(5px);
            padding: 3rem 2rem;
        }
        
        /* Feature cards */
        .feature-card {
            text-align: center;
            padding: 2rem 1.5rem;
            background: rgba(102, 126, 234, 0.08);
            border-radius: 15px;
            border: 1.5px solid rgba(102, 126, 234, 0.25);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .feature-card:hover {
            background: rgba(102, 126, 234, 0.15);
            border-color: rgba(102, 126, 234, 0.5);
            transform: translateY(-8px);
            box-shadow: 0 12px 30px rgba(102, 126, 234, 0.2);
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            background-color: rgba(255, 255, 255, 0.1);
            color: #e0e0e0;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: rgba(102, 126, 234, 0.3) !important;
            color: #ffffff !important;
        }
        
        /* Caption text */
        .stCaption {
            color: #b0b0b0 !important;
        }
        
        /* Code blocks */
        code {
            background-color: rgba(0, 0, 0, 0.3) !important;
            color: #f8f8f2 !important;
            padding: 0.2em 0.4em;
            border-radius: 3px;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

    # Header with gradient
    st.markdown("""
        <div class="main-header">
            <h1>✈️ AI Trip Planner</h1>
            <p>Discover amazing destinations with AI-powered recommendations, real-time weather, and travel options</p>
        </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        openweather_api_key = os.getenv("OPENWEATHER_API_KEY", "").strip()
        
        # API Key Status (compact)
        if not google_api_key or not openweather_api_key:
            st.error("⚠️ **API Keys Required**")
            with st.expander("🔑 Setup Instructions"):
                if not google_api_key:
                    st.markdown("**❌ GOOGLE_API_KEY** not set")
                else:
                    st.markdown("**✅ GOOGLE_API_KEY** set")
                
                if not openweather_api_key:
                    st.markdown("**❌ OPENWEATHER_API_KEY** not set")
                else:
                    st.markdown("**✅ OPENWEATHER_API_KEY** set")
                
                st.markdown("---")
                st.markdown("**Get API Keys:**")
                st.markdown("- [Gemini API](https://makersuite.google.com/app/apikey)")
                st.markdown("- [OpenWeather](https://openweathermap.org/api)")
            st.markdown("---")
        
        # Trip Details Section
        st.markdown("### 🗺️ Trip Details")
        
        col1, col2 = st.columns(2)
        with col1:
            origin_city = st.text_input("📍 Origin", value="Delhi", help="Your starting city")
        with col2:
            destination_city = st.text_input("🎯 Destination", value="Tokyo", help="Where you want to go")
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("📅 Start Date", value=date.today())
        with col2:
            num_days = st.number_input("📆 Days", min_value=1, max_value=14, value=3, step=1)

        month_label = st.selectbox(
            "🗓️ Month",
            options=["(use exact dates only)", "May", "June", "July", "August", "September", "October"],
            index=1,
        )
        month_label_clean = None if month_label.startswith("(") else month_label

        preferences = st.text_area(
            "💭 Preferences",
            value="Mix of culture, food, and light sightseeing. Prefer central area hotels.",
            help="Add any constraints or preferences (budget, interests, pace, etc.)",
            height=100,
        )

        st.markdown("---")
        
        # Model Selection (simplified)
        st.markdown("### ⚙️ Settings")
        model_name = st.selectbox(
            "🤖 AI Model",
            options=[
                "gemini-2.5-flash",
                "gemini-flash-latest",
                "gemini-2.0-flash",
                "gemini-2.5-pro",
                "gemini-pro-latest",
            ],
            index=0,
            help="Flash models have better free tier quotas",
        )
        
        if "pro" in model_name.lower() and "preview" in model_name.lower():
            st.caption("⚠️ Pro preview models may have quota limits")

        st.markdown("---")
        
        # Generate Button
        run_button = st.button("✨ Generate Trip Plan", type="primary", disabled=not google_api_key or not openweather_api_key, use_container_width=True)
        
        if not google_api_key or not openweather_api_key:
            st.caption("⚠️ Set API keys to enable trip planning")

    # Main content area
    if not run_button:
        # Welcome section
        st.markdown("""
        <div class="welcome-section" style="text-align: center; padding: 3rem 1.5rem; background: rgba(102, 126, 234, 0.12); border-radius: 15px; margin: 2rem 0; border: 2px solid rgba(102, 126, 234, 0.3);">
            <h2 style="color: #9d8df1; margin-bottom: 1rem; font-weight: 800; font-size: 2rem;">🌍 Ready to Plan Your Next Adventure?</h2>
            <p style="font-size: 1.15rem; color: #e0e0e0; max-width: 700px; margin: 0 auto; line-height: 1.6;">
                Fill in your trip details in the sidebar and click <strong style="color: #9d8df1;">"Generate Trip Plan"</strong> to get personalized recommendations!
            </p>
            <div style="display: flex; justify-content: center; align-items: center; gap: 1.5rem; margin-top: 2.5rem; flex-wrap: nowrap; overflow-x: auto; padding: 0 1rem;">
                <div class="feature-card" style="padding: 1.5rem 1.75rem; background: rgba(102, 126, 234, 0.15); border-radius: 12px; border: 1px solid rgba(102, 126, 234, 0.4); transition: all 0.3s ease; flex-shrink: 0; display: flex; flex-direction: column; align-items: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🏛️</div>
                    <div style="font-weight: 700; margin-top: 0.25rem; color: #e0e0e0; font-size: 0.9rem;">City Overview</div>
                </div>
                <div class="feature-card" style="padding: 1.5rem 1.75rem; background: rgba(102, 126, 234, 0.15); border-radius: 12px; border: 1px solid rgba(102, 126, 234, 0.4); transition: all 0.3s ease; flex-shrink: 0; display: flex; flex-direction: column; align-items: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🌤️</div>
                    <div style="font-weight: 700; margin-top: 0.25rem; color: #e0e0e0; font-size: 0.9rem;">Weather</div>
                </div>
                <div class="feature-card" style="padding: 1.5rem 1.75rem; background: rgba(102, 126, 234, 0.15); border-radius: 12px; border: 1px solid rgba(102, 126, 234, 0.4); transition: all 0.3s ease; flex-shrink: 0; display: flex; flex-direction: column; align-items: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">✈️</div>
                    <div style="font-weight: 700; margin-top: 0.25rem; color: #e0e0e0; font-size: 0.9rem;">Travel Options</div>
                </div>
                <div class="feature-card" style="padding: 1.5rem 1.75rem; background: rgba(102, 126, 234, 0.15); border-radius: 12px; border: 1px solid rgba(102, 126, 234, 0.4); transition: all 0.3s ease; flex-shrink: 0; display: flex; flex-direction: column; align-items: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🎭</div>
                    <div style="font-weight: 700; margin-top: 0.25rem; color: #e0e0e0; font-size: 0.9rem;">Attractions</div>
                </div>
                <div class="feature-card" style="padding: 1.5rem 1.75rem; background: rgba(102, 126, 234, 0.15); border-radius: 12px; border: 1px solid rgba(102, 126, 234, 0.4); transition: all 0.3s ease; flex-shrink: 0; display: flex; flex-direction: column; align-items: center;">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">📅</div>
                    <div style="font-weight: 700; margin-top: 0.25rem; color: #e0e0e0; font-size: 0.9rem;">Day Plan</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    if run_button:
        if not origin_city.strip() or not destination_city.strip():
            st.error("⚠️ Please provide both an origin city and a destination city.")
            return

        # Show progress with better messaging
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.info("🌍 Gathering weather data...")
        progress_bar.progress(20)
        
        try:
            status_text.info("✈️ Searching travel options...")
            progress_bar.progress(40)
            
            status_text.info("🤖 AI is crafting your personalized trip plan...")
            progress_bar.progress(60)
            
            plan_markdown = run_planner(
                city=destination_city.strip(),
                origin=origin_city.strip(),
                start_date=start_date,
                num_days=int(num_days),
                month_label=month_label_clean,
                preferences=preferences.strip(),
                model_name=model_name,
            )
            
            progress_bar.progress(100)
            status_text.success("✅ Trip plan generated successfully!")
            progress_bar.empty()
            status_text.empty()
            
        except Exception as e:  # noqa: BLE001
            progress_bar.empty()
            status_text.empty()
            
            error_msg = str(e)
            
            # Check for quota/rate limit errors
            if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg or "quota" in error_msg.lower():
                st.error("⚠️ **Quota/Rate Limit Exceeded**")
                
                st.warning(
                    f"""
                    **The model `{model_name}` has exceeded its free tier quota.**
                    
                    **💡 Quick Fix:** Switch to a Flash model in the sidebar:
                    - `gemini-2.5-flash` ⭐ Recommended
                    - `gemini-flash-latest`
                    - `gemini-2.0-flash`
                    
                    Or wait a few minutes and try again.
                    """
                )
                
                # Show retry delay if mentioned in error
                if "retry" in error_msg.lower() or "seconds" in error_msg.lower():
                    import re
                    retry_match = re.search(r'(\d+\.?\d*)\s*seconds?', error_msg, re.IGNORECASE)
                    if retry_match:
                        retry_seconds = float(retry_match.group(1))
                        st.info(f"⏱️ Suggested retry delay: {int(retry_seconds)} seconds")
            else:
                st.error("❌ **Failed to generate trip plan**")
                with st.expander("Error Details"):
                    st.code(error_msg)
            return

        # Display trip plan in a styled container
        st.markdown("""
            <div class="trip-plan-container">
        """, unsafe_allow_html=True)
        
        st.markdown("## ✈️ Your Personalized Trip Plan")
        st.markdown("---")
        
        # Render the markdown content
        st.markdown(plan_markdown)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Add download/share options
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                label="📥 Download Plan",
                data=plan_markdown,
                file_name=f"trip_plan_{destination_city}_{start_date}.md",
                mime="text/markdown",
            )
        with col2:
            if st.button("🔄 Plan Another Trip"):
                st.rerun()
        with col3:
            st.markdown("")


if __name__ == "__main__":
    main()
