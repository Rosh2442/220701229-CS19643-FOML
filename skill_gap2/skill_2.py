import streamlit as st
import pandas as pd
import requests
import io
from PyPDF2 import PdfReader
import sqlite3
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# ---- Job Roles and Skills ----
job_roles = {
    "Data Scientist": ["Python", "Pandas", "NumPy", "Machine Learning", "Statistics", "SQL"],
    "Web Developer": ["HTML", "CSS", "JavaScript", "React", "Node.js"],
    "Cybersecurity Analyst": ["Networking", "Linux", "Python", "Firewalls", "Risk Assessment"],
    "AI Engineer": ["Python", "TensorFlow", "Deep Learning", "Data Structures", "Math"],
    "Mobile App Developer": ["Flutter", "Dart", "Java", "Kotlin", "Firebase"],
    "Frontend Developer": ["HTML", "CSS", "JavaScript", "UI/UX"],
    "Backend Developer": ["Python", "Django", "SQL", "REST APIs"],
    "Full Stack Developer": ["JavaScript", "React", "Node.js", "MongoDB"],
    "DevOps Engineer": ["Docker", "Kubernetes", "CI/CD", "AWS"],
    "Cloud Engineer": ["AWS", "Azure", "Terraform", "Networking"],
    "Data Engineer": ["SQL", "ETL", "Apache Spark", "Airflow"],
    "UI/UX Designer": ["Figma", "Wireframing", "Prototyping"],
    "Game Developer": ["Unity", "C#", "Game Design"],
    "Product Manager": ["Agile", "Roadmapping", "Analytics"],
    "Business Analyst": ["Excel", "Power BI", "SQL"],
    "Machine Learning Engineer": ["Python", "Scikit-Learn", "TensorFlow", "MLOps"],
    "AR/VR Developer": ["Unity", "ARKit", "ARCore"],
    "Blockchain Developer": ["Solidity", "Smart Contracts", "Web3.js"],
    "QA Engineer": ["Selenium", "Test Cases", "Postman"],
    "Prompt Engineer": ["LLMs", "Prompt Design", "ChatGPT", "LangChain"],
    "Technical Writer": ["Markdown", "Documentation", "APIs", "Git"],
    "Data Analyst": ["Excel", "SQL", "Power BI", "Data Visualization"],
    "Network Engineer": ["Cisco", "Routing", "Switching", "Firewalls"],
    "Site Reliability Engineer (SRE)": ["Monitoring", "Kubernetes", "Incident Management"],
    "Systems Administrator": ["Linux", "Bash", "Monitoring", "Automation"],
    "Ethical Hacker": ["Penetration Testing", "Metasploit", "Burp Suite", "Python"],
    "Digital Marketing Specialist": ["SEO", "Google Analytics", "Social Media", "Content Marketing"],
    "Robotics Engineer": ["ROS", "C++", "Sensors", "Microcontrollers"]
}

def train_role_prediction_model(job_roles):
    roles = list(job_roles.keys())
    skills = [job_roles[role] for role in roles]
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(skills)
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, roles)
    return model, mlb

def predict_roles(skills, model, mlb):
    input_vector = mlb.transform([skills])
    distances, indices = model.kneighbors(input_vector, n_neighbors=3)
    predictions = [model.classes_[i] for i in indices[0]]
    return predictions

model, mlb = train_role_prediction_model(job_roles)


conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)''')
cursor.execute('''CREATE TABLE IF NOT EXISTS profiles (username TEXT, role TEXT, skills TEXT)''')
cursor.execute('''CREATE TABLE IF NOT EXISTS progress (username TEXT, role TEXT, skill TEXT, status TEXT)''')


def register_user(username, password):
    cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
    conn.commit()

def login_user(username, password):
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    return cursor.fetchone()

def get_courses(skill):
    return f"[Top {skill} courses on Coursera](https://www.coursera.org/search?query={skill.replace(' ', '%20')})"

def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    return text

def extract_skills_from_text(text, all_skills):
    return [skill for skill in all_skills if skill.lower() in text.lower()]

def extract_skills_from_description(description, all_skills):
    return [skill for skill in all_skills if skill.lower() in description.lower()]

def show_learning_path(skill):
    learning_path = {
        "Python": ["Variables", "Loops", "Functions", "OOP", "Projects"],
        "Pandas": ["DataFrames", "Cleaning", "GroupBy", "Visualization"],
        "Power BI": ["Basic Charts", "DAX", "Reports", "Dashboards"]
    }
    if skill in learning_path:
        steps = learning_path[skill]
        st.markdown(f"**🧽 Learning Path for {skill}:**")
        for i, step in enumerate(steps, 1):
            st.markdown(f"{i}. {step}")
    else:
        st.markdown(f"🔍 Explore general resources for **{skill}**")

# ---- UI Setup ----
st.set_page_config(page_title="Skill-Gap Visualizer", layout="wide")
st.title("🚀Skill-Gap Visualizer")
st.markdown("""
<style>
/* --- Professional Gradient Background (Steel Blue to Light Gray) --- */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: linear-gradient(180deg, #e0eafc, #cfdef3);
    background-attachment: fixed;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* --- Sidebar Container Styling --- */
section[data-testid="stSidebar"] {
    background: transparent;
    padding: 2rem 1rem;
    display: flex;
    align-items: center;
    justify-content: center;
}

/* --- Glassmorphism Panel Styling --- */
section[data-testid="stSidebar"] > div:first-child {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 20px;
    backdrop-filter: blur(10px);
    padding: 25px 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    color: #333;
    width: 100%;
    max-height: 600px;
    overflow-y: auto;
}

/* --- Sidebar Header --- */
section[data-testid="stSidebar"] h2 {
    font-size: 26px;
    font-weight: 700;
    color: #003366;
    margin-bottom: 1rem;
}

/* --- Radio Buttons --- */
section[data-testid="stSidebar"] .stRadio div[role='radiogroup'] > div {
    background-color: #f0f4f8;
    color: #003366 !important;
    padding: 10px 14px;
    border-radius: 6px;
    margin-bottom: 6px;
    cursor: pointer;
}

/* --- Selected Radio Option --- */
section[data-testid="stSidebar"] .stRadio div[aria-checked='true'] {
    background-color: #cfe2f3;
    font-weight: 600;
}

/* --- Text Inputs --- */
section[data-testid="stSidebar"] input {
    background-color: #f5f7fa;
    color: #333;
    padding: 10px;
    border-radius: 8px;
    border: 1px solid #d1d9e6;
    margin-top: 8px;
}

section[data-testid="stSidebar"] input::placeholder {
    color: #888;
}

/* --- Submit Button --- */
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(to right, #1d3557, #457b9d);
    color: white;
    font-weight: bold;
    padding: 10px 20px;
    border-radius: 8px;
    margin-top: 16px;
    width: 100%;
    border: none;
    transition: background 0.3s ease;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(to right, #16324f, #3b6d8c);
}

/* --- Remove White Outline on Click --- */
section[data-testid="stSidebar"] .stButton > button:focus {
    outline: none;
    box-shadow: none;
}
</style>
""", unsafe_allow_html=True)

# ---- Sidebar ----
with st.sidebar:
    st.header("🔐 Access Panel")
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        auth_choice = st.radio("Choose Option", ["Login", "Register"])
        input_username = st.text_input("Username")
        input_password = st.text_input("Password", type="password")
        auth_button = st.button("Submit")

        if auth_button:
            if auth_choice == "Login":
                user = login_user(input_username, input_password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.username = input_username
                    st.success("✅ Logged in successfully!")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
            else:
                register_user(input_username, input_password)
                st.success("✅ Account created! Please log in.")
    else:
        st.success(f"👋 Welcome back, {st.session_state.username}")
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            del st.session_state.username
            st.rerun()

        st.markdown("## 📁 Dashboard")
        role_list = cursor.execute("SELECT DISTINCT role FROM progress WHERE username=?", (st.session_state.username,)).fetchall()
        for (role,) in role_list:
            if st.button(role):
                st.session_state.view_dashboard = True
                st.session_state.selected_role = role
                st.rerun()

# ---- Main App ----
if st.session_state.get("logged_in"):
    username = st.session_state.username

    if st.session_state.get("view_dashboard") and st.session_state.get("selected_role"):
        role = st.session_state.selected_role
        st.subheader(f"📘 Progress for {role}")
        skills = cursor.execute("SELECT skill, status FROM progress WHERE username=? AND role=?", (username, role)).fetchall()
        for i, (skill, status) in enumerate(skills):
            col1, col2 = st.columns([2, 3])
            with col1:
                status_icon = {
                    "Not Started": "🔴",
                    "In Progress": "🟡",
                    "Completed": "🟢"
                }.get(status, "⚪")
                st.markdown(f"{status_icon} **{skill}**")
            with col2:
                new_status = st.selectbox(
                    "Update Status",
                    ["Not Started", "In Progress", "Completed"],
                    index=["Not Started", "In Progress", "Completed"].index(status),
                    key=f"edit_{username}_{role}_{skill}_{i}"
                )

        # ✅ Update immediately if changed
                if new_status != status:
                    cursor.execute(
                        "UPDATE progress SET status=? WHERE username=? AND role=? AND skill=?",
                        (new_status, username, role, skill)
                    )
                    conn.commit()
                    st.success(f"✅ '{skill}' in {role} updated to '{new_status}'")
                    st.rerun()
        if st.button("🔙 Back to Homepage"):
            del st.session_state["view_dashboard"]
            del st.session_state["selected_role"]
            st.rerun()

        st.stop()

    st.subheader("📄 Upload Resume or ✍️ Enter Skills")
    uploaded_resume = st.file_uploader("Upload Resume (PDF)", type=['pdf'])
    manual_input = st.text_input("Or enter your skills manually (comma-separated)")

    st.subheader("📋 Paste a Job Description")
    job_description = st.text_area("Or paste a job description to extract required skills")

    all_skills = sorted({skill for skills in job_roles.values() for skill in skills})

    current_skills = set()
    if uploaded_resume:
        text = extract_text_from_pdf(uploaded_resume)
        extracted_skills = extract_skills_from_text(text, all_skills)
        st.success(f"Extracted Skills: {', '.join(extracted_skills)}")
        current_skills = set(extracted_skills)
    elif manual_input:
        current_skills = set(skill.strip().title() for skill in manual_input.split(","))

    if job_description:
        jd_skills = extract_skills_from_description(job_description, all_skills)
        st.success(f"Extracted required skills from JD: {', '.join(jd_skills)}")
        current_skills.update(jd_skills)

    st.subheader("🎯 Career Path Selection")
    selected_roles = st.multiselect("Manually select roles you're interested in", list(job_roles.keys()))

    if current_skills:
        if st.checkbox("🔮 Suggest roles based on my skills (ML-powered)"):
            predicted_roles = predict_roles(list(current_skills), model, mlb)
            st.success(f"✅ Suggested Roles: {', '.join(predicted_roles)}")
            selected_roles = list(set(selected_roles) | set(predicted_roles))

    if st.button("🔍 Analyze & Recommend"):
        if not current_skills or not selected_roles:
            st.warning("⚠️ Please enter skills and choose roles.")
        else:
            st.success(f"🔎 Analysis for **{username}**")

            results = {}
            for role in selected_roles:
                required = set(job_roles[role])
                matched = current_skills & required
                missing = required - current_skills
                results[role] = (len(matched), len(missing))

                cursor.execute("INSERT INTO profiles (username, role, skills) VALUES (?, ?, ?)",
                               (username, role, ", ".join(current_skills)))
                conn.commit()

                for skill in missing:
                    cursor.execute("INSERT OR IGNORE INTO progress (username, role, skill, status) VALUES (?, ?, ?, ?)",
                                   (username, role, skill, "Not Started"))
                    conn.commit()

                st.markdown(f"---\n### 💼 {role}")
                st.markdown(f"✅ **Known Skills:** {', '.join(matched)}")
                st.markdown(f"❌ **Skills to Learn:** {', '.join(missing)}")
                st.markdown("📘 **Learning Resources:**")
                for skill in missing:
                    st.markdown(f"- {skill}: {get_courses(skill)}")
                    show_learning_path(skill)

            st.subheader("📊 Skill Match Overview")
            roles = list(results.keys())
            matches = [results[r][0] for r in roles]
            gaps = [results[r][1] for r in roles]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(roles, matches, label='Matched', color='#00cc66')
            ax.bar(roles, gaps, bottom=matches, label='Missing', color='#ff6666')
            ax.set_ylabel('Skills')
            ax.set_title('Skill Match vs Gap')
            ax.legend()
            st.pyplot(fig)

else:
    st.info("👈 Please login from the sidebar to use the app.")
    st.markdown("### 👋 Welcome to Skill-Gap Visualizer!")
    st.markdown("To begin, please **log in** or **register** using the panel on the **left sidebar**.")
    st.markdown("This tool helps you visualize your skills and find gaps for your dream career path.")