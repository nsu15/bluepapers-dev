import os
import pandas as pd
import re
from flask import Flask, render_template, request
from openai import OpenAI
from rapidfuzz import fuzz

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Load datasets
rv_data = pd.read_csv("blue_papers.csv")
us_locations_df = pd.read_csv("us_locations.csv")
facilities_df = pd.read_csv("rehab_facilities.csv")

# Clean column headers
us_locations_df.columns = us_locations_df.columns.str.lower().str.strip()
facilities_df.columns = facilities_df.columns.str.lower().str.strip()

# Flask app
app = Flask(__name__)

# Normalize location inputs
def get_location_from_input(user_input):
    user_input = user_input.lower()
    location_found = None
    for _, row in us_locations_df.iterrows():
        city = row.get("city", "").lower()
        state = row.get("state", "").lower()
        abbr = row.get("abbreviation", "").lower()
        if city and fuzz.partial_ratio(city, user_input) > 90:
            location_found = f"{row['city']}, {row['state']}"
            break
        elif abbr and abbr in user_input:
            location_found = row['state']
            break
        elif state and state in user_input:
            location_found = row['state']
            break
    return location_found

# Get local facilities
def find_facilities(location, topic):
    if not location or not topic:
        return []
    facilities = []
    for _, row in facilities_df.iterrows():
        if fuzz.partial_ratio(location.lower(), row.get("location", "").lower()) > 85:
            if fuzz.partial_ratio(topic.lower(), row.get("type", "").lower()) > 70:
                facilities.append(f"{row['name']} – {row['address']} – {row['phone']}")
    return facilities

# Handle different assistant modes
def generate_ai_response(mode, user_input, location=None, topic=None):
    system_messages = {
        "hope": "You are a supportive housing assistant helping people find affordable places to live. The user is searching for housing options that meet their needs. For now, only RV data is available. Give a warm, clear explanation of options they might consider based on the input. Also describe any steps they should take next, like who to call or how to apply. Respond like a helpful, encouraging guide.",
        "everyday_answers": "You are a compassionate assistant helping users solve difficult life situations related to housing, travel, mobility, or rebuilding their life. Provide actionable steps and encouragement based on the user's input. Keep your response practical and hopeful.",
        "talk_to_me": "The user is dealing with a life challenge (such as loneliness, addiction, or mental health stress). Respond like a kind and encouraging friend — offer gentle advice and affirm their strength. Also recommend 2–3 types of nearby places that could help — like support centers, rehab clinics, or housing offices. Be conversational and supportive, not robotic or deflective.",
    }

    system_prompt = system_messages.get(mode, system_messages["everyday_answers"])
    full_prompt = f"{system_prompt}: {user_input}"

    if location and topic:
        facilities = find_facilities(location, topic)
        if facilities:
            facilities_text = "Here are some places that might help:\n" + "\n".join(facilities[:3])
            full_prompt += f"\n\n{facilities_text}"

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ]
    )

    return response.choices[0].message.content

# RV search
def search_rvs(user_input):
    price_match = re.search(r"under \$?([\d,]+)", user_input, re.IGNORECASE)
    state_match = re.search(r"in ([A-Za-z\s]+)", user_input, re.IGNORECASE)

    price_limit = int(price_match.group(1).replace(",", "")) if price_match else None
    state = state_match.group(1).strip() if state_match else None

    matches = rv_data.copy()
    if price_limit:
        matches = matches[matches["AVERAGE_PRICE"] <= price_limit]
    if state:
        matches = matches[matches["STATE"].str.lower() == state.lower()]
    matches = matches.sort_values(by="AVERAGE_PRICE", ascending=True).head(10)
    return matches.to_dict("records")

# Flask route
# ... [unchanged imports and logic above remain the same]

@app.route("/", methods=["GET", "POST"])
def index():
    ai_explanation = ""
    ai_results = []
    spoken_text = ""
    formatted_text = ""

    if request.method == "POST":
        user_input = request.form.get("query", "")
        mode = request.form.get("mode", "hope")

        location = get_location_from_input(user_input)
        topic = "addiction" if any(word in user_input.lower() for word in ["drinking", "alcohol", "rehab", "drug", "addiction"]) else "help"

        if mode in ["hope", "everyday_answers"]:
            ai_results = search_rvs(user_input)

        ai_explanation = generate_ai_response(mode, user_input, location, topic)
        spoken_text = ai_explanation.replace("\n", " ").replace("**", "")
        formatted_text = ai_explanation.replace("\n", "<br>").replace("**", "")

    return render_template("index.html",
                           ai_explanation=formatted_text,
                           ai_results=ai_results,
                           spoken_text=spoken_text)

if __name__ == "__main__":
    app.run(debug=True)
