import streamlit as st
import fitz
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI HR Copilot",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# SKILLS DATABASE
# ============================================================

SKILL_LIST = [
    "Python",
    "Java",
    "C++",
    "C",
    "SQL",
    "MySQL",
    "PostgreSQL",
    "HTML",
    "CSS",
    "JavaScript",
    "Machine Learning",
    "Deep Learning",
    "PyTorch",
    "TensorFlow",
    "Scikit-learn",
    "Pandas",
    "NumPy",
    "Docker",
    "Git",
    "GitHub",
    "Linux",
    "AWS",
    "Azure",
    "Power BI",
    "Excel",
    "Flask",
    "Django",
    "FastAPI",
    "React",
    "Node.js",
    "MongoDB",
    "NLP",
    "Computer Vision",
    "REST API"
]


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #f4f7fc;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 {
        color: #172554;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #dfe6f1;
        padding: 18px;
        border-radius: 16px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.04);
    }

    div[data-testid="stExpander"] {
        background: white;
        border: 1px solid #dfe6f1;
        border-radius: 14px;
        margin-bottom: 10px;
    }

    div[data-testid="stFileUploader"] {
        background: white;
        border-radius: 15px;
        padding: 10px;
    }

    .match-skill {
        display: inline-block;
        background: #dcfce7;
        color: #166534;
        border: 1px solid #86efac;
        padding: 6px 11px;
        border-radius: 20px;
        margin: 3px;
        font-weight: 600;
    }

    .missing-skill {
        display: inline-block;
        background: #fee2e2;
        color: #991b1b;
        border: 1px solid #fecaca;
        padding: 6px 11px;
        border-radius: 20px;
        margin: 3px;
        font-weight: 600;
    }

    .required-skill {
        display: inline-block;
        background: #ede9fe;
        color: #5b21b6;
        border: 1px solid #c4b5fd;
        padding: 6px 11px;
        border-radius: 20px;
        margin: 3px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.title("🤖 AI HR Copilot")

st.write(
    "AI-powered resume screening, candidate ranking, "
    "skill-gap analysis and HR decision support."
)

st.divider()


# ============================================================
# LOAD SENTENCE TRANSFORMER
# ============================================================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load AI model: {error}")
    st.stop()


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_text(uploaded_file):

    uploaded_file.seek(0)

    pdf_bytes = uploaded_file.read()

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page in document:
        pages.append(page.get_text("text"))

    document.close()

    text = "\n".join(pages).strip()

    return text


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    text = text.lower()

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# SKILL DETECTION
# ============================================================

def skill_present(skill, text):

    text_lower = normalize_text(text)
    skill_lower = skill.lower()

    # Special cases
    if skill_lower == "c++":
        return "c++" in text_lower

    if skill_lower == "c":
        return bool(
            re.search(
                r"(?<![a-zA-Z+#])c(?![a-zA-Z+#])",
                text_lower
            )
        )

    if skill_lower == "sql":
        return bool(
            re.search(
                r"\bsql\b",
                text_lower
            )
        )

    if skill_lower == "git":
        return bool(
            re.search(
                r"\bgit\b",
                text_lower
            )
        )

    return skill_lower in text_lower


def find_skills(text):

    found = []

    for skill in SKILL_LIST:

        if skill_present(skill, text):
            found.append(skill)

    return list(dict.fromkeys(found))


# ============================================================
# EMAIL
# ============================================================

def extract_email(text):

    match = re.search(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        text
    )

    if match:
        return match.group(0)

    return "Not found"


# ============================================================
# PHONE
# ============================================================

def extract_phone(text):

    patterns = [
        r"(?:\+91[\s-]?)?[6-9]\d{9}",
        r"(?:\+91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}"
    ]

    for pattern in patterns:

        match = re.search(pattern, text)

        if match:
            return match.group(0)

    return "Not found"


# ============================================================
# NAME
# ============================================================

def extract_name(text, filename):

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    unwanted_words = [
        "resume",
        "curriculum",
        "email",
        "phone",
        "mobile",
        "linkedin",
        "github",
        "objective",
        "summary",
        "profile"
    ]

    for line in lines[:12]:

        lower = line.lower()

        if any(word in lower for word in unwanted_words):
            continue

        if "@" in line:
            continue

        if re.search(r"\d{6,}", line):
            continue

        if len(line) > 50:
            continue

        words = line.split()

        if 2 <= len(words) <= 5:
            return line

    # fallback to filename
    clean_name = re.sub(
        r"^\d+[_\-\s]*",
        "",
        filename
    )

    clean_name = re.sub(
        r"\.pdf$",
        "",
        clean_name,
        flags=re.IGNORECASE
    )

    clean_name = clean_name.replace("_", " ")

    return clean_name.strip()


# ============================================================
# EDUCATION
# ============================================================

def extract_education(text):

    education_keywords = [
        "B.Tech",
        "B.E",
        "Bachelor of Technology",
        "Bachelor of Engineering",
        "BCA",
        "MCA",
        "M.Tech",
        "B.Sc",
        "M.Sc",
        "MBA",
        "Computer Science",
        "Information Technology"
    ]

    found = []

    text_lower = text.lower()

    for education in education_keywords:

        if education.lower() in text_lower:
            found.append(education)

    return list(dict.fromkeys(found))


# ============================================================
# EXPERIENCE
# ============================================================

def extract_experience(text):

    patterns = [
        r"\b\d+(?:\.\d+)?\+?\s*(?:years|year|yrs|yr)\s+(?:of\s+)?experience\b",
        r"\bexperience\s*[:\-]?\s*\d+(?:\.\d+)?\+?\s*(?:years|year|yrs|yr)\b",
        r"\b\d+\+?\s*(?:months|month)\s+(?:of\s+)?experience\b"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            return match.group(0)

    if re.search(
        r"\bfresher\b",
        text,
        re.IGNORECASE
    ):
        return "Fresher"

    return "Not specified"


# ============================================================
# PROJECT EXTRACTION
# ============================================================

def extract_projects(text):

    keywords = [
        "project",
        "developed",
        "built",
        "implemented",
        "created"
    ]

    projects = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        if len(line) > 250:
            continue

        lower = line.lower()

        if any(word in lower for word in keywords):
            projects.append(line)

    return list(dict.fromkeys(projects))[:5]


# ============================================================
# CERTIFICATION EXTRACTION
# ============================================================

def extract_certifications(text):

    keywords = [
        "certification",
        "certificate",
        "certified",
        "coursera",
        "udemy",
        "nptel"
    ]

    certifications = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        if len(line) > 250:
            continue

        lower = line.lower()

        if any(word in lower for word in keywords):
            certifications.append(line)

    return list(dict.fromkeys(certifications))[:5]


# ============================================================
# CANDIDATE EXTRACTION
# ============================================================

def extract_candidate_details(text, filename):

    return {
        "Name": extract_name(text, filename),
        "Email": extract_email(text),
        "Phone": extract_phone(text),
        "Skills": find_skills(text),
        "Education": extract_education(text),
        "Experience": extract_experience(text),
        "Projects": extract_projects(text),
        "Certifications": extract_certifications(text)
    }


# ============================================================
# SEMANTIC SIMILARITY
# ============================================================

def calculate_semantic_score(job_description, resume_text):

    embeddings = model.encode(
        [job_description, resume_text],
        convert_to_numpy=True
    )

    similarity = cosine_similarity(
        embeddings[0].reshape(1, -1),
        embeddings[1].reshape(1, -1)
    )[0][0]

    # cosine similarity -> percentage
    score = float(similarity) * 100

    return round(
        max(0, min(score, 100)),
        2
    )


# ============================================================
# SKILL GAP
# ============================================================

def analyze_skill_gap(job_description, candidate_skills):

    required_skills = find_skills(job_description)

    candidate_lower = {
        skill.lower()
        for skill in candidate_skills
    }

    matching = []
    missing = []

    for skill in required_skills:

        if skill.lower() in candidate_lower:
            matching.append(skill)
        else:
            missing.append(skill)

    return required_skills, matching, missing


# ============================================================
# FINAL SCORE
# ============================================================

def calculate_final_score(
    semantic_score,
    required_skills,
    matching_skills
):

    if required_skills:

        skill_score = (
            len(matching_skills)
            / len(required_skills)
        ) * 100

        # 60% semantic + 40% skill coverage
        final_score = (
            semantic_score * 0.60
            + skill_score * 0.40
        )

    else:
        skill_score = 0
        final_score = semantic_score

    return (
        round(max(0, min(final_score, 100)), 2),
        round(skill_score, 2)
    )


# ============================================================
# HR DECISION ENGINE
# ============================================================

def generate_hr_decision(score):

    if score >= 75:
        return "Strongly Recommend"

    elif score >= 60:
        return "Recommend"

    elif score >= 40:
        return "Consider"

    else:
        return "Reject"


# ============================================================
# STRENGTHS / WEAKNESSES
# ============================================================

def generate_strengths(matching_skills):

    if matching_skills:

        return (
            "Strong alignment in: "
            + ", ".join(matching_skills)
            + "."
        )

    return (
        "No major required technical skills "
        "were directly matched."
    )


def generate_weaknesses(missing_skills):

    if missing_skills:

        return (
            "Missing or not clearly identified: "
            + ", ".join(missing_skills)
            + "."
        )

    return (
        "No major required skill gaps "
        "were identified."
    )


# ============================================================
# SCORE EXPLANATION
# ============================================================

def generate_reasons(
    semantic_score,
    skill_score,
    matching_skills,
    missing_skills
):

    reasons = []

    reasons.append(
        f"Semantic resume-to-job similarity: "
        f"{semantic_score:.1f}%"
    )

    reasons.append(
        f"Required skill coverage: "
        f"{skill_score:.1f}%"
    )

    for skill in matching_skills:
        reasons.append(f"✓ {skill} matches")

    for skill in missing_skills:
        reasons.append(f"✗ {skill} missing")

    return reasons


# ============================================================
# BADGES
# ============================================================

def show_badges(skills, badge_type):

    if not skills:
        st.caption("None identified")
        return

    if badge_type == "match":
        css = "match-skill"

    elif badge_type == "missing":
        css = "missing-skill"

    else:
        css = "required-skill"

    html = ""

    for skill in skills:

        safe_skill = (
            str(skill)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        html += (
            f'<span class="{css}">'
            f'{safe_skill}'
            f'</span>'
        )

    st.markdown(
        html,
        unsafe_allow_html=True
    )


# ============================================================
# USER INPUT
# ============================================================

st.header("📥 Resume Screening")

left, right = st.columns(2)


with left:

    st.subheader("💼 Job Description")

    job_description = st.text_area(
        "Paste the complete Job Description",
        height=250,
        placeholder=(
            "Example: We are looking for a Machine Learning "
            "Engineer with Python, Machine Learning, PyTorch, "
            "SQL, Docker and Linux skills..."
        )
    )


with right:

    st.subheader("📄 Candidate Resumes")

    uploaded_resumes = st.file_uploader(
        "Upload Resume PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_resumes:

        st.success(
            f"{len(uploaded_resumes)} resume(s) uploaded."
        )

        for file in uploaded_resumes:
            st.caption(f"📄 {file.name}")


analyze = st.button(
    "🚀 Analyze Candidates",
    type="primary",
    use_container_width=True
)


# ============================================================
# RUN ANALYSIS
# ============================================================

if analyze:

    if not job_description.strip():

        st.error(
            "Please enter the Job Description."
        )

        st.stop()


    if not uploaded_resumes:

        st.error(
            "Please upload at least one resume."
        )

        st.stop()


    results = []

    failed_files = []

    progress = st.progress(0)

    status = st.empty()


    for index, resume in enumerate(uploaded_resumes):

        status.info(
            f"Analyzing {resume.name}..."
        )

        try:

            # ----------------------------------------------
            # EXTRACT PDF
            # ----------------------------------------------

            resume_text = extract_pdf_text(resume)


            if len(resume_text.strip()) < 20:

                raise ValueError(
                    "Very little or no readable text "
                    "was extracted from this PDF."
                )


            # ----------------------------------------------
            # CANDIDATE DETAILS
            # ----------------------------------------------

            details = extract_candidate_details(
                resume_text,
                resume.name
            )


            # ----------------------------------------------
            # SEMANTIC SCORE
            # ----------------------------------------------

            semantic_score = calculate_semantic_score(
                job_description,
                resume_text
            )


            # ----------------------------------------------
            # SKILLS
            # ----------------------------------------------

            (
                required_skills,
                matching_skills,
                missing_skills
            ) = analyze_skill_gap(
                job_description,
                details["Skills"]
            )


            # ----------------------------------------------
            # FINAL SCORE
            # ----------------------------------------------

            final_score, skill_score = (
                calculate_final_score(
                    semantic_score,
                    required_skills,
                    matching_skills
                )
            )


            # ----------------------------------------------
            # DECISION
            # ----------------------------------------------

            recommendation = (
                generate_hr_decision(
                    final_score
                )
            )


            strengths = generate_strengths(
                matching_skills
            )

            weaknesses = generate_weaknesses(
                missing_skills
            )


            reasons = generate_reasons(
                semantic_score,
                skill_score,
                matching_skills,
                missing_skills
            )


            # ----------------------------------------------
            # STORE RESULT
            # ----------------------------------------------

            results.append({
                "Candidate": details["Name"],
                "Email": details["Email"],
                "Phone": details["Phone"],
                "Skills": details["Skills"],
                "Education": details["Education"],
                "Experience": details["Experience"],
                "Projects": details["Projects"],
                "Certifications": details["Certifications"],
                "Semantic Score": semantic_score,
                "Skill Score": skill_score,
                "Match Score (%)": final_score,
                "Matching Skills": matching_skills,
                "Missing Skills": missing_skills,
                "Strengths": strengths,
                "Weaknesses": weaknesses,
                "Recommendation": recommendation,
                "Reasons": reasons,
                "Filename": resume.name
            })


        except Exception as error:

            failed_files.append({
                "Filename": resume.name,
                "Error": str(error)
            })


        progress.progress(
            (index + 1) / len(uploaded_resumes)
        )


    progress.empty()
    status.empty()


    # ========================================================
    # DISPLAY PROCESSING ERRORS
    # ========================================================

    if failed_files:

        st.warning(
            f"{len(failed_files)} resume(s) could not "
            "be processed."
        )

        with st.expander(
            "⚠️ View processing errors"
        ):

            for failed in failed_files:

                st.error(
                    f'{failed["Filename"]}: '
                    f'{failed["Error"]}'
                )


    if not results:

        st.error(
            "None of the uploaded resumes could be analyzed. "
            "Please check the processing errors above."
        )

        st.stop()


    # ========================================================
    # SORT AND RANK
    # ========================================================

    results = sorted(
        results,
        key=lambda x: x["Match Score (%)"],
        reverse=True
    )


    for rank, candidate in enumerate(
        results,
        start=1
    ):

        candidate["Rank"] = rank


    # ========================================================
    # JOB DESCRIPTION SUMMARY
    # ========================================================

    required_skills = find_skills(
        job_description
    )


    shortlisted = sum(
        1
        for candidate in results
        if candidate["Recommendation"] in [
            "Strongly Recommend",
            "Recommend"
        ]
    )


    average_match = (
        sum(
            candidate["Match Score (%)"]
            for candidate in results
        )
        / len(results)
    )


    st.success(
        "✅ Candidate analysis completed."
    )


    st.header(
        "📋 Job Description Summary"
    )


    st.subheader(
        "Required Skills"
    )

    show_badges(
        required_skills,
        "required"
    )


    metric1, metric2, metric3 = st.columns(3)


    with metric1:

        st.metric(
            "👥 Candidates Screened",
            len(results)
        )


    with metric2:

        st.metric(
            "✅ Shortlisted",
            shortlisted
        )


    with metric3:

        st.metric(
            "📊 Average Match",
            f"{average_match:.1f}%"
        )


    # ========================================================
    # RANKED CANDIDATES
    # ========================================================

    st.header(
        "🏆 Ranked Candidates"
    )


    for candidate in results:

        score = candidate["Match Score (%)"]

        recommendation = candidate[
            "Recommendation"
        ]


        if recommendation == "Strongly Recommend":
            icon = "🟢"

        elif recommendation == "Recommend":
            icon = "🔵"

        elif recommendation == "Consider":
            icon = "🟡"

        else:
            icon = "🔴"


        title = (
            f'#{candidate["Rank"]} '
            f'{candidate["Candidate"]} '
            f'| {score:.1f}% '
            f'| {icon} {recommendation}'
        )


        with st.expander(title):

            score_col, info_col = st.columns(
                [1, 2]
            )


            # ----------------------------------------------
            # SCORE PANEL
            # ----------------------------------------------

            with score_col:

                st.metric(
                    "🎯 Match Score",
                    f"{score:.1f}%"
                )

                st.progress(
                    min(
                        max(int(score), 0),
                        100
                    )
                )

                st.write(
                    f"**{icon} {recommendation}**"
                )

                st.caption(
                    f'Semantic score: '
                    f'{candidate["Semantic Score"]:.1f}%'
                )

                st.caption(
                    f'Skill coverage: '
                    f'{candidate["Skill Score"]:.1f}%'
                )


            # ----------------------------------------------
            # CONTACT
            # ----------------------------------------------

            with info_col:

                st.subheader(
                    candidate["Candidate"]
                )

                st.write(
                    f'📧 **Email:** '
                    f'{candidate["Email"]}'
                )

                st.write(
                    f'📞 **Phone:** '
                    f'{candidate["Phone"]}'
                )

                st.write(
                    f'💼 **Experience:** '
                    f'{candidate["Experience"]}'
                )

                st.caption(
                    f'File: '
                    f'{candidate["Filename"]}'
                )


            # ----------------------------------------------
            # TABS
            # ----------------------------------------------

            tab1, tab2, tab3, tab4 = st.tabs(
                [
                    "🎯 Skills",
                    "👤 Profile",
                    "🧠 Score Explanation",
                    "⚖️ HR Decision"
                ]
            )


            # SKILLS
            with tab1:

                st.subheader(
                    "✅ Matching Skills"
                )

                show_badges(
                    candidate["Matching Skills"],
                    "match"
                )


                st.subheader(
                    "❌ Missing Skills"
                )

                show_badges(
                    candidate["Missing Skills"],
                    "missing"
                )


                st.subheader(
                    "🛠️ All Detected Skills"
                )

                show_badges(
                    candidate["Skills"],
                    "required"
                )


            # PROFILE
            with tab2:

                st.subheader(
                    "🎓 Education"
                )

                if candidate["Education"]:

                    for education in candidate[
                        "Education"
                    ]:

                        st.write(
                            f"• {education}"
                        )

                else:

                    st.write(
                        "Education not detected."
                    )


                st.subheader(
                    "🚀 Projects"
                )

                if candidate["Projects"]:

                    for project in candidate[
                        "Projects"
                    ]:

                        st.write(
                            f"• {project}"
                        )

                else:

                    st.write(
                        "Projects not detected."
                    )


                st.subheader(
                    "🏅 Certifications"
                )

                if candidate[
                    "Certifications"
                ]:

                    for certification in candidate[
                        "Certifications"
                    ]:

                        st.write(
                            f"• {certification}"
                        )

                else:

                    st.write(
                        "Certifications not detected."
                    )


            # EXPLANATION
            with tab3:

                st.subheader(
                    "Why was this score given?"
                )


                for reason in candidate[
                    "Reasons"
                ]:

                    if reason.startswith("✓"):

                        st.success(reason)

                    elif reason.startswith("✗"):

                        st.warning(reason)

                    else:

                        st.info(reason)


                st.write(
                    "**Final score calculation:**"
                )

                st.code(
                    "Final Match Score = "
                    "(Semantic Similarity × 60%) + "
                    "(Skill Coverage × 40%)"
                )


            # HR DECISION
            with tab4:

                st.subheader(
                    "💪 Strengths"
                )

                st.success(
                    candidate["Strengths"]
                )


                st.subheader(
                    "⚠️ Weaknesses"
                )

                st.warning(
                    candidate["Weaknesses"]
                )


                st.subheader(
                    "🎯 Recommendation"
                )


                if recommendation in [
                    "Strongly Recommend",
                    "Recommend"
                ]:

                    st.success(
                        f"{icon} {recommendation}"
                    )

                elif recommendation == "Consider":

                    st.warning(
                        f"{icon} {recommendation}"
                    )

                else:

                    st.error(
                        f"{icon} {recommendation}"
                    )


    # ========================================================
    # TOP CANDIDATE
    # ========================================================

    best_candidate = results[0]


    st.header(
        "⭐ Top Recommended Candidate"
    )


    top1, top2, top3 = st.columns(3)


    with top1:

        st.metric(
            "Candidate",
            best_candidate["Candidate"]
        )


    with top2:

        st.metric(
            "Match Score",
            f'{best_candidate["Match Score (%)"]:.1f}%'
        )


    with top3:

        st.metric(
            "Recommendation",
            best_candidate["Recommendation"]
        )


    # ========================================================
    # FINAL HR REPORT
    # ========================================================

    st.header(
        "📊 Final HR Report"
    )


    report_data = []


    for candidate in results:

        report_data.append({

            "Rank":
                candidate["Rank"],

            "Candidate":
                candidate["Candidate"],

            "Email":
                candidate["Email"],

            "Phone":
                candidate["Phone"],

            "Match %":
                candidate["Match Score (%)"],

            "Recommendation":
                candidate["Recommendation"],

            "Matching Skills":
                ", ".join(
                    candidate["Matching Skills"]
                ),

            "Missing Skills":
                ", ".join(
                    candidate["Missing Skills"]
                ),

            "Strengths":
                candidate["Strengths"],

            "Weaknesses":
                candidate["Weaknesses"]

        })


    report_df = pd.DataFrame(
        report_data
    )


    st.dataframe(
        report_df,
        hide_index=True,
        use_container_width=True
    )


    # ========================================================
    # DOWNLOAD REPORT
    # ========================================================

    csv_data = report_df.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        "⬇️ Download Final HR Report",
        data=csv_data,
        file_name="AI_HR_Copilot_Report.csv",
        mime="text/csv",
        use_container_width=True
    )