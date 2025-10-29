# ------------------------------------------------------------
# Mini College Admission Chatbot using NLTK + Streamlit
# Auto Clear Input + Course-wise Detailed Explanations
# ------------------------------------------------------------
import os
import ssl
import nltk
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# SSL fix for nltk downloads (helps if SSL errors occur)
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

# ensure a local nltk_data folder is considered (optional)
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download("punkt", quiet=True)

# ---------------------- INTENTS DATA ----------------------
data = {
    "greeting": {
        "examples": ["Hi", "Hello", "Hey", "Good morning", "Good evening"],
        "responses": [
            "Hello! Welcome to ABC College. How can I assist you today?",
            "Hi there! Are you looking for admission details?"
        ],
    },
    "courses_offered": {
        "examples": [
            "What courses do you offer?",
            "Available courses?",
            "UG programs?",
            "Engineering branches?"
        ],
        "responses": [
            "We offer B.E in CSE, ECE, Mechanical, and Civil Engineering.",
            "Our UG programs include Computer Science, Electronics, Mechanical and Civil Engineering."
        ],
    },
    "admission_process": {
        "examples": ["How to apply for admission?", "Admission process", "How can I join your college?"],
        "responses": [
            "You can apply online through our official website under the admissions section.",
            "Admission is usually based on entrance exam scores followed by counseling."
        ],
    },
    "fees": {
        "examples": ["What is the fee structure?", "Annual fees?", "Tuition fees details"],
        "responses": [
            "The annual fee varies by course, starting from around ₹80,000 per year.",
            "Please check our official admission portal for detailed fee structure by program."
        ],
    },
    "scholarship": {
        "examples": ["Do you provide scholarships?", "Any scholarship available?"],
        "responses": [
            "Yes, scholarships are available for meritorious and economically weaker students.",
            "We offer government and management scholarships based on eligibility criteria."
        ],
    },
    "hostel": {
        "examples": ["Do you have hostel facilities?", "Hostel details", "Is hostel available?"],
        "responses": [
            "Yes, we provide separate hostel facilities for boys and girls with basic amenities.",
            "Hostel includes Wi-Fi, mess, 24/7 security and medical facilities."
        ],
    },
    "placements": {
        "examples": ["Placement record", "What about placements?", "Company visits?"],
        "responses": [
            "Our placement cell organizes campus drives with reputed recruiters every year.",
            "Placement percentages vary by department; check our placement reports for details."
        ],
    },
    "cutoff": {
        "examples": ["Cutoff marks", "What is the cutoff for CSE?", "Last year cutoff?"],
        "responses": [
            "Cutoffs vary each year and by category — please refer to the official admissions page for accurate cutoffs.",
        ],
    },
    "thanks": {
        "examples": ["Thanks", "Thank you", "Thanks a lot"],
        "responses": ["You're welcome!", "Happy to help 🙂"]
    },
    "goodbye": {
        "examples": ["Bye", "See you", "Goodbye", "Thanks, bye"],
        "responses": ["Goodbye! Feel free to reach out for any more queries.", "Thanks for visiting — all the best!"]
    },
    "default": {
        "examples": [""],
        "responses": ["I'm sorry, I didn't understand. Could you please rephrase your question?"]
    }
}

# ---------------------- Course detailed descriptions (explicit) ----------------------
course_details = {
    "cse": """**CSE (Computer Science & Engineering)**  
Focus: Programming, data structures, algorithms, software development, databases, operating systems, AI, ML, data science, cloud computing.  
Career paths: Software engineer, data scientist, ML engineer, devops, product engineer.  
Why choose CSE: High industry demand, strong placement opportunities, wide scope in startups and product companies.""",

    "ece": """**ECE (Electronics & Communication Engineering)**  
Focus: Analog & digital electronics, communication systems, signal processing, microprocessors, embedded systems, VLSI.  
Career paths: Embedded systems engineer, hardware design engineer, telecom engineer, IoT developer.  
Why choose ECE: Good for hardware + software crossover roles, IoT and telecom sectors.""" ,

    "mech": """**Mechanical Engineering**  
Focus: Thermodynamics, mechanics, manufacturing processes, CAD/CAM, materials, machine design.  
Career paths: Design engineer, production/plant engineer, automotive engineer, R&D.  
Why choose Mechanical: Strong core engineering fundamentals, opportunities in manufacturing and automotive industries.""",

    "civil": """**Civil Engineering**  
Focus: Structural analysis, construction technology, geotechnical engineering, transportation engineering, environmental engineering.  
Career paths: Structural engineer, site engineer, project manager, urban planner.  
Why choose Civil: Key role in infrastructure projects and construction sector, long-term demand in public and private projects."""
}

# ---------------------- TRAIN MODEL ----------------------
X_train, y_train = [], []
for intent, intent_data in data.items():
    for ex in intent_data["examples"]:
        if ex.strip():
            X_train.append(ex)
            y_train.append(intent)

# Vectorize + Train
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X_train)
model = LogisticRegression(max_iter=300)
model.fit(X_vectorized, y_train)

# ---------------------- Helper: detect course keyword ----------------------
def detect_course_query(text: str):
    t = text.lower()
    # check common ways user may refer to courses
    if "cse" in t or "computer" in t or "computer science" in t:
        return "cse"
    if "ece" in t or "electronics" in t or "electrical" in t:
        return "ece"
    if "mech" in t or "mechanical" in t or "mechanics" in t:
        return "mech"
    if "civil" in t or "civil engineering" in t:
        return "civil"
    return None

# ---------------------- Chatbot response function (improved) ----------------------
def chatbot_response(user_input: str):
    # 1) If the user explicitly asks about a course, return the detailed course description
    course_key = detect_course_query(user_input)
    if course_key:
        return course_details[course_key]

    # 2) otherwise use the trained intent classifier
    if not user_input.strip():
        return "Please enter a message."
    X_test = vectorizer.transform([user_input])
    intent = model.predict(X_test)[0]
    responses = data.get(intent, data["default"])["responses"]
    return random.choice(responses)

# ---------------------- STREAMLIT UI ----------------------
st.set_page_config(page_title="College Admission Chatbot", page_icon="🎓", layout="centered")
st.title("🎓 ABC College — Admission Enquiry Chatbot")
st.write("💬 Ask about courses (e.g., 'Tell me about CSE'), admissions, fees, hostel, placements, and more.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input form (clear_on_submit ensures box is cleared on submit)
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your message:", key="chat_input")
    submit = st.form_submit_button("Send")

if submit:
    if user_input.strip():
        # get bot reply (course-specific or intent-based)
        bot_reply = chatbot_response(user_input)
        # append to history
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", bot_reply))
        # rerun so the input clears immediately (clear_on_submit=True already clears the widget,
        # rerun ensures the UI immediately updates to show cleared input)
        st.rerun()
    else:
        st.warning("Please enter a message!")

# Display chat history (simple vertical list)
if st.session_state.chat_history:
    st.markdown("### 🗨️ Conversation")
    for sender, message in st.session_state.chat_history:
        emoji = "🧑‍💻" if sender == "You" else "🤖"
        # Using markdown for richer formatting (course descriptions already contain bold / line breaks)
        st.markdown(f"**{emoji} {sender}:** {message}")

st.markdown("---")
st.markdown("Developed by **SK** — For ABC College Admission Demo")
