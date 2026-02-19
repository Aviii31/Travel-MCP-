#!/usr/bin/env python3
"""
Script to check which Gemini models you have access to.

Run:
    export GOOGLE_API_KEY="your-api-key"
    python check_gemini_models.py
"""

import os
import sys

try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai package not installed.")
    print("Install it with: pip install google-generativeai")
    sys.exit(1)


def list_available_models():
    """List all available Gemini models for the current API key."""
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable is not set.")
        print("Set it with: export GOOGLE_API_KEY='your-api-key'")
        sys.exit(1)

    # Configure the API
    genai.configure(api_key=api_key)

    print("Fetching available Gemini models...")
    print("=" * 60)

    try:
        # List all models
        models = genai.list_models()
        
        # Filter for Gemini models
        gemini_models = []
        for model in models:
            if 'gemini' in model.name.lower():
                gemini_models.append(model)

        if not gemini_models:
            print("No Gemini models found. This might indicate:")
            print("1. Your API key doesn't have access to Gemini models")
            print("2. There's an issue with your API key")
            print("3. The API endpoint is not accessible")
            return

        print(f"\nFound {len(gemini_models)} Gemini model(s):\n")
        
        # Group by model family
        flash_models = []
        pro_models = []
        other_models = []
        
        for model in gemini_models:
            name = model.name.replace('models/', '')
            if 'flash' in name.lower():
                flash_models.append((name, model))
            elif 'pro' in name.lower():
                pro_models.append((name, model))
            else:
                other_models.append((name, model))

        if flash_models:
            print("📱 Flash Models (Faster, lighter):")
            for name, model in sorted(flash_models):
                supported = []
                if 'generateContent' in model.supported_generation_methods:
                    supported.append("✅ generateContent")
                print(f"  • {name}")
                if supported:
                    print(f"    {', '.join(supported)}")
            print()

        if pro_models:
            print("🚀 Pro Models (More capable):")
            for name, model in sorted(pro_models):
                supported = []
                if 'generateContent' in model.supported_generation_methods:
                    supported.append("✅ generateContent")
                print(f"  • {name}")
                if supported:
                    print(f"    {', '.join(supported)}")
            print()

        if other_models:
            print("🔧 Other Gemini Models:")
            for name, model in sorted(other_models):
                supported = []
                if 'generateContent' in model.supported_generation_methods:
                    supported.append("✅ generateContent")
                print(f"  • {name}")
                if supported:
                    print(f"    {', '.join(supported)}")
            print()

        print("=" * 60)
        print("\n💡 Tip: Use the model name (without 'models/' prefix) in your Streamlit app.")
        print("   Example: If you see 'models/gemini-2.0-flash-exp', use 'gemini-2.0-flash-exp'")

    except Exception as e:
        print(f"Error fetching models: {e}")
        print("\nTroubleshooting:")
        print("1. Verify your GOOGLE_API_KEY is correct")
        print("2. Check if your API key has access to Gemini models")
        print("3. Ensure you have internet connectivity")
        sys.exit(1)


if __name__ == "__main__":
    list_available_models()
