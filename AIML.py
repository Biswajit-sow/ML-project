import os
import json
import gradio as gr
from dotenv import load_dotenv
from uuid import uuid4
import re
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from PIL import Image

# --- Core LangChain components ---
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# --- Document Loading components ---
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader

# --- 1. SET UP THE ENVIRONMENT ---
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]="true"# tracing we have to kept it as true so it is automatically going to do the tracing  With respect to any code that write 
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY") # Langchain API key actually  help us to know that where the  entire monitoring result needs to be stored right so that dashboard you will be able  to see the entire monitoring result will be over here 
os.environ["LANGCHAIN_PROJECT"] = "AI-ML Project"
# --- All backend Python functions are UNCHANGED ---

def create_gap_analysis_chart(candidate_profile, job_requirements):
    if not candidate_profile or not job_requirements: return None
    labels = list(candidate_profile.keys())
    candidate_stats = list(candidate_profile.values())
    job_stats = list(job_requirements.values())
    abbr = {"Technical Skills": "Tech", "Tools & Technologies": "Tools", "Relevant Experience": "Exp", "Soft Skills": "Soft"}
    job_label_parts = [f"{abbr.get(k, k)}: {v}" for k, v in job_requirements.items()]
    job_label = f"Job Requirement ({', '.join(job_label_parts)})"
    candidate_label_parts = [f"{abbr.get(k, k)}: {v}" for k, v in candidate_profile.items()]
    candidate_label = f"Your Profile ({', '.join(candidate_label_parts)})"
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    candidate_stats += candidate_stats[:1]
    job_stats += job_stats[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)
    plt.xticks(angles[:-1], labels, color='grey', size=12)
    plt.yticks([25, 50, 75, 100], ["", "", "", ""], color="grey", size=7)
    plt.ylim(0, 100)
    ax.plot(angles, job_stats, 'o--', linewidth=2, color='lightcoral', label=job_label)
    ax.fill(angles, job_stats, 'lightcoral', alpha=0.25)
    ax.plot(angles, candidate_stats, 'o-', linewidth=2, color='deepskyblue', label=candidate_label)
    ax.fill(angles, candidate_stats, 'deepskyblue', alpha=0.4)
    plt.title('Skill Gap Analysis', size=20, color='white', y=1.12)
    legend = ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), facecolor='#404040', edgecolor='white', labelcolor='white', title='📊 Profile Key', fontsize='medium', fancybox=True, shadow=True)
    plt.setp(legend.get_title(), color='skyblue', weight='bold', fontsize='large')
    filepath = f"gap_chart_{uuid4()}.png"
    plt.savefig(filepath, transparent=True, dpi=100, bbox_inches='tight')
    plt.close(fig)
    return filepath

import textwrap
from matplotlib.font_manager import FontProperties
import matplotlib.patheffects as path_effects

def create_learning_roadmap_chart(roadmap_data):
    if not roadmap_data or 'roadmap' not in roadmap_data:
        return None
    topics = [item['topic'] for item in roadmap_data['roadmap']]
    durations = [item['duration_weeks'] for item in roadmap_data['roadmap']]
    stages = [item['stage'] for item in roadmap_data['roadmap']]
    milestones = [item.get('milestone') for item in roadmap_data['roadmap']]
    unique_stages = sorted(list(set(stages)), key=stages.index)
    colors = plt.colormaps.get_cmap('viridis')(np.linspace(0, 1, len(unique_stages)))
    stage_colors = {stage: colors[i] for i, stage in enumerate(unique_stages)}
    fig, ax = plt.subplots(figsize=(25, len(topics) * 1.5 + 3))
    y_pos = np.arange(len(topics))
    start_time = 0
    try:
        emoji_font = FontProperties(family=['Segoe UI Emoji', 'Arial'])
    except:
        emoji_font = FontProperties(family=['Arial'])
    for i in range(len(topics)):
        bar_color = stage_colors[stages[i]]
        topic_len_factor = len(topics[i]) / 15.0
        min_width = 3.5 + topic_len_factor
        display_duration = max(durations[i], min_width)
        ax.barh(y_pos[i], display_duration, left=start_time, align='center',
                      height=0.6, color=bar_color, edgecolor='black', linewidth=1.5)
        r, g, b, _ = bar_color
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = 'white' if luminance < 0.5 else 'black'
        wrap_width = max(int(display_duration * 2.8), 15)
        wrapped_topic = textwrap.fill(topics[i], width=wrap_width)
        ax.text(start_time + display_duration/2, y_pos[i], wrapped_topic,
                va='center', ha='center', color=text_color, fontweight='bold', fontsize=10.5)
        if milestones[i] and milestones[i].strip():
            wrapped_milestone = textwrap.fill(f"⭐ {milestones[i]}", width=70)
            milestone_text = ax.text(start_time, y_pos[i] + 0.45, wrapped_milestone,
                                     va='top', ha='left', color='gold',
                                     fontsize=16, fontweight='bold', fontproperties=emoji_font)
            milestone_text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'),
                                             path_effects.Normal()])
        start_time += durations[i]
    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel('Timeline in Weeks', fontsize=15, fontweight='bold')
    ax.set_title(roadmap_data.get('chart_title', 'Learning Roadmap'), fontsize=22, fontweight='bold', pad=35)
    patches = [mpatches.Patch(color=color, label=stage) for stage, color in stage_colors.items()]
    legend = ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
    legend.get_frame().set_edgecolor('black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.set_xlim(0, start_time + 2)
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    filepath = f"roadmap_chart_{uuid4()}.png"
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return filepath

def extract_json_from_string(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    return None

def format_analysis_for_display(analysis_json):
    if "error" in analysis_json: return f"### Analysis Error\n- {analysis_json['error']}"
    match_score = analysis_json.get('match_score', 'N/A')
    feedback = analysis_json.get('feedback', 'No feedback provided.')
    keywords_matched = analysis_json.get('keywords_matched', [])
    keywords_missing = analysis_json.get('keywords_missing', [])
    requirements_analysis = analysis_json.get('requirements_analysis', [])
    interview_questions = analysis_json.get('interview_questions', [])
    output_str = f"## 📝 In-Depth Career Analysis\n\n"
    output_str += f"### **Overall Profile Match: {match_score}**\n\n"
    output_str += f"**Analyst's Verdict:** *{feedback}*\n\n---\n\n"
    if keywords_matched or keywords_missing:
        output_str += "### **Keyword Analysis** 🔑\n"
        output_str += "**Keywords from the JD found in your resume:**\n"
        output_str += f"> {', '.join(keywords_matched) if keywords_matched else 'None found.'}\n\n"
        output_str += "**Critical keywords missing from your resume:**\n"
        output_str += f"> {', '.join(keywords_missing) if keywords_missing else 'None identified.'}\n\n---\n\n"
    if requirements_analysis:
        output_str += "### **Requirement-by-Requirement Breakdown** ✅\n"
        for req in requirements_analysis:
            status = req.get('status', 'N/A')
            verdict_emoji = "✔️" if status == "Match" else "⚠️" if status == "Partial Match" else "❌"
            output_str += f"- **{verdict_emoji} Requirement:** *{req.get('requirement', '')}*\n"
            output_str += f"  - **Verdict:** {status}\n"
            output_str += f"  - **Justification:** {req.get('justification', '')}\n"
        output_str += "\n---\n\n"
    if interview_questions:
        output_str += "### **Potential Interview Questions** 🎙️\n"
        output_str += "Based on this analysis, be prepared to answer questions like these:\n"
        for i, q in enumerate(interview_questions, 1):
            output_str += f"{i}. {q}\n"
        output_str += "\n"
    return output_str

def load_document_text(file_path):
    if not file_path: return ""
    _, extension = os.path.splitext(file_path)
    if extension.lower() == ".pdf": loader = PyMuPDFLoader(file_path)
    elif extension.lower() == ".docx": loader = Docx2txtLoader(file_path)
    else: raise ValueError(f"Unsupported file type: {extension}")
    documents = loader.load()
    return "\n".join(doc.page_content for doc in documents)

def run_career_advisor_analysis(resume_file, job_description, model_name):
    if "GROQ_API_KEY" not in os.environ or not os.environ["GROQ_API_KEY"]:
        return "LLM not configured. Check your GROQ_API_KEY.", "", None
    if not resume_file or not job_description:
        return "Please upload a resume and provide a job description.", "", None
    try:
        llm = ChatGroq(temperature=0.1, model_name=model_name)
    except Exception as e:
        return f"Error initializing the LLM: {e}", "", None
    advisor_template = """
    You are a highly precise and analytical AI Career Advisor. Your primary task is to conduct a detailed and deterministic comparison between a user's resume and a given job description. Your analysis should strictly assess whether the skills, experiences, and qualifications mentioned in the resume align with the requirements of the job description. Only if there is a strong match should a high accuracy score be provided; otherwise, assign a lower score with justified reasoning. Be firm and objective in your evaluation—do not inflate scores for partial matches. If the match is weak, clearly identify the specific areas where the candidate needs improvement. Your ideality demands a high standard of accuracy, consistency, and constructive feedback. Maintain a tone of professional strictness while guiding the user toward enhancing their profile effectively.
    Resume Text: --- {resume_text} ---
    Job Description: --- {job_description} ---
    Your response MUST be a single, well-formed JSON object with the following exact keys:
    1.  "candidate_profile": A dictionary rating the CANDIDATE on "Technical Skills", "Tools & Technologies", "Relevant Experience", and "Soft Skills" out of 100.
    2.  "job_requirements": A dictionary rating the IDEAL candidate for the JOB on the same 4 categories.
    3.  "match_score": A percentage string based on your analysis. Be consistent.
    4.  "keywords_matched": A list of important keywords from the job description that ARE present in the resume.
    5.  "keywords_missing": A list of important keywords from the job description that ARE NOT present in the resume.
    6.  "requirements_analysis": A list of objects. For each main requirement in the job description, create an object with three keys: "requirement" (the requirement text), "status" ("Match", "Partial Match", or "No Match"), and "justification" (a brief explanation for your verdict).
    7.  "interview_questions": A list of 3-4 potential interview questions based on the gaps and strengths identified.
    8.  "feedback": A concise, overall summary and verdict.
    Provide ONLY the JSON object.
    """
    advisor_prompt = PromptTemplate.from_template(advisor_template)
    advisor_chain = LLMChain(llm=llm, prompt=advisor_prompt)
    chart_path = None
    try:
        resume_text = load_document_text(resume_file.name)
        raw_response = advisor_chain.run({"resume_text": resume_text, "job_description": job_description})
        json_str = extract_json_from_string(raw_response)
        if not json_str:
            analysis_json = {"error": "Model's response did not contain valid JSON."}
        else:
            analysis_json = json.loads(json_str)
            candidate_data = analysis_json.get("candidate_profile")
            job_data = analysis_json.get("job_requirements")
            if candidate_data and job_data:
                chart_path = create_gap_analysis_chart(candidate_data, job_data)
        formatted_output = format_analysis_for_display(analysis_json)
        missing_keywords = analysis_json.get("keywords_missing", [])
        learning_goal = f"I need to learn about these missing keywords for a job: {', '.join(missing_keywords)}." if missing_keywords else "My resume seems to cover all keywords. I'd like to deepen my project experience."
        return formatted_output, learning_goal, chart_path
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return f"An unexpected error occurred: {e}", "", None

session_memories = {}

def get_mentor_chain(session_id, model_name):
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatGroq(temperature=0.7, model_name=model_name)
    mentor_prompt_template = """You are a helpful and friendly Smart Academic Mentor. Your primary goal is to help users break down their learning goals into a structured, actionable plan.
    **If any user asks your name, you must always respond that Hey , NEXUS here . Clearly state that you are an AI Career Advisor and a Smart Academic Mentor designed to assist with career guidance, skill development, academic planning, and personalized learning journeys.** Maintain consistency in this identity across all responses.

    **SPECIAL INSTRUCTIONS FOR VISUALIZATION:**
    1.  Analyze the user's request. If the user explicitly asks for a "chart," "roadmap," "visual," "graph," or "timeline" for their learning plan, you MUST generate a special JSON object for visualization.
    2.  This JSON object must be placed at the very end of your response, enclosed in 
json ...
 tags.
    3.  The JSON object must have this exact structure (NOTE: The curly braces {{ and }} are escaped for the template):
        '''json
        {{
          "is_chart_request": true,
          "chart_title": "A descriptive title for the chart (e.g., 'Full-Stack Developer Roadmap')",
          "roadmap": [
            {{
              "stage": "Name of the learning phase (e.g., 'Phase 1: Foundations')",
              "topic": "Specific skill or topic to learn (e.g., 'HTML & CSS Basics')",
              "duration_weeks": <integer, estimated number of weeks>,
              "milestone": "A brief project or goal for this topic (e.g., 'Build a personal portfolio page')"
            }},
            {{
              "stage": "Phase 2: Backend",
              "topic": "Node.js & Express",
              "duration_weeks": 4,
              "milestone": "Build a REST API"
            }}
          ]
        }}
        '''
    4.  Even when generating the JSON, you must ALSO provide your normal, conversational, and encouraging step-by-step learning plan in Markdown format BEFORE the JSON block. The JSON is for the system; the text is for the user.
    5.  If the user is simply chatting or asking a question that is not explicitly requesting a full learning roadmap visual, do not generate any JSON or diagram. Only respond conversationally. When generating a chart, ensure it is of high visibility and professional quality: the chart must have no overlapping between numbers, text, boxes, or lines. Every part of the chart should be easily readable without requiring the user to zoom in. The output must be crisp, non-blurry, and designed for maximum readability and a premium appearance.

        --Additionally, if the user says something like "I am learning...","I want to learn... followed by a topic (for example, "I am learning deep learning" or "I started learning web development","I want to Learn astology" and oters), first provide helpful and accurate information or guidance on that topic.
        -- After providing that information, ask the user whether they would like to see a visual diagram or chart for better understanding.
            --If the user responds yes, then generate and present the visual.
            --If the user responds no, do not generate the visual.

    🔴 ABSOLUTE RULE — NO EXCEPTIONS:
        --If the user does not explicitly request a visual diagram or chart, you must NOT generate or include any visual by default.
        --Here is an ultra-strict rule for generating visual diagrams:
            **The assistant must ensure that every visual diagram generated meets the highest standards of clarity, readability, and design excellence.
            ** All text within the diagram must be rendered in ultra-high quality, using large, bold, and legible fonts that require no zooming or squinting. The layout must be clean, spacious, and professionally structured—absolutely no overlapping of text, labels, boxes, or arrows is permitted under any circumstance. Every element of the diagram must be clearly visible and aesthetically aligned to create a premium, polished appearance. If there is even a slight risk of blurriness, compression artifacts, or cramped spacing, the assistant must **not generate** the diagram until all issues are corrected. The final output must look like a presentation-ready infographic—flawless, easy to read, and instantly understandable. Any diagram that fails to meet this ultra-high-quality standard must be considered unacceptable.

        --Do not assume, suggest, or preemptively include any diagram unless the user clearly states that they want one. This applies even if the topic typically involves visuals or the user has previously asked for diagrams. Every new request must be judged independently, and visuals must only be provided upon direct request.
        --Violation of this rule breaks the expected behavior of the assistant and is not allowed under any circumstance.
    6.If any user asks you to play a quiz on any topic, you must generate a quiz with multiple-choice questions presented in A, B, C, D options.

        --Ask the question and wait for the user to respond before revealing the correct answer.
        --If the user gives the correct answer, respond with a message like "✅ Correct! Here's the explanation:" followed by a brief explanation.
        --If the user gives the wrong answer, respond with a message like "❌ That's not correct. The right answer is [Correct Option]. Here's why:" and then provide the explanation.

    🔴 Strict Rule: Never include the answer with the question. Always wait for the user to respond first.
    7. Always retain memory of previously generated charts and user-specific instructions.
        --If a user previously discussed or requested a chart—whether it was a roadmap, skill gap analysis, learning flow, or any other visual—you must remember the details, structure, and preferences provided earlier. This includes the topic, visual format, font style, padding, spacing, color scheme, and any design-related feedback.
        --If the user refers back to a "previous chart" or asks to “update,” “improve,” or “regenerate” a chart without re-specifying all details, recall and reuse the latest known preferences and topic context to deliver a consistent and intelligent follow-up. Only ask clarifying questions if something has changed or is unclear.
    🔴 ABSOLUTE RULE — NO EXCEPTIONS:The assistant must always remember and apply the user's previous instructions, preferences, and decisions throughout the entire session without exception. This includes remembering which types of charts or visualizations the user approved or rejected, as well as any formatting or design choices previously discussed. The assistant is not allowed to forget, ignore, or alter any past instruction unless the user explicitly updates or cancels it. If the user previously mentioned whether they want visuals or not, that decision must be strictly followed in future responses. Maintaining accurate memory of prior interactions is essential, and any failure to do so is considered a serious violation of assistant behavior.
    Previous conversation:
    {chat_history}
    New question: {input}
    Your Response:
    """
    mentor_prompt = PromptTemplate.from_template(mentor_prompt_template)
    return LLMChain(llm=llm, prompt=mentor_prompt, memory=session_memories[session_id], verbose=True)

def mentor_chat_response(message, history, session_id, model_name):
    if "GROQ_API_KEY" not in os.environ or not os.environ["GROQ_API_KEY"]:
        history.append([message, "Error: LLM not configured. Check your GROQ_API_KEY."])
        return history
    history.append([message, None])
    try:
        chain = get_mentor_chain(session_id, model_name)
        raw_response = chain.predict(input=message)
    except Exception as e:
        history[-1][1] = f"An error occurred while communicating with the LLM: {e}"
        return history
    text_response = raw_response
    chart_path = None
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
        text_response = raw_response.replace(json_match.group(0), "").strip()
        try:
            roadmap_json = json.loads(json_str)
            if roadmap_json.get("is_chart_request"):
                chart_path = create_learning_roadmap_chart(roadmap_json)
        except Exception as e:
            print(f"Error processing roadmap JSON or creating chart: {e}")
            text_response += "\n\n_(Sorry, I had trouble generating the visual roadmap chart.)_"
    history[-1][1] = text_response
    if chart_path:
        history.append([None, (chart_path,)])
    return history

import speech_recognition as sr
# --- NEW: Function to transcribe audio to text ---
def transcribe_audio(audio_path):
    if not audio_path:
        return ""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            # Using Google's free web API for transcription
            text = recognizer.recognize_google(audio_data)
            print(f"Transcription: '{text}'")
            return text
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            return "Sorry, I could not understand the audio. Please try again."
        except sr.RequestError as e:
            print(f"Could not request results from Google service; {e}")
            return f"API Error: Could not connect to the speech recognition service."

def switch_ui_mode(mode_selection):
    is_advisor_visible = (mode_selection == "1. AI Career Advisor")
    is_mentor_visible = (mode_selection == "2. Smart Academic Mentor")
    return {
        advisor_ui_container: gr.update(visible=is_advisor_visible),
        mentor_ui_container: gr.update(visible=is_mentor_visible)
    }

# --- MODIFIED: CSS now handles the gap between columns ---
custom_css = """
/* --- [Theme: "NexusUI Futurist"] --- */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&family=Roboto:wght@400;500&display=swap');

:root {
    --nexus-bg-dark: #0d1117;
    --nexus-bg-deep-blue: #020B26;
    --nexus-border-color: #30363d;
    --nexus-primary-accent: #00e0ff;
    --nexus-secondary-accent: #c951ff;
    --nexus-label-bg: #3b82f6;
    --nexus-text-color: #c9d1d9;
    --nexus-text-color-light: #f0f6fc;
    --font-main: 'Roboto', sans-serif;
    --font-title: 'Orbitron', sans-serif;
}

body {
    background: var(--nexus-bg-dark);
    color: var(--nexus-text-color);
    font-family: var(--font-main);
}
.gradio-container { background: none; }

/* App Title */
#gradio-title {
    font-family: var(--font-title);
    font-size: 3em;
    color: var(--nexus-text-color-light);
    text-shadow: 0 0 8px var(--nexus-primary-accent), 0 0 16px var(--nexus-secondary-accent);
    padding-top: 24px;
    padding-left: 20px;
    padding-bottom: 20px;
    margin-bottom: 10px;
}

/* ALL text headings and labels (excluding markdowns with different class) */
h1, h2, h3, h4, h5, h6, p, span, .gradio-label, .gradio-markdown, label, .gradio-dropdown label {
    font-family: var(--font-title) !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px;
}

.gradio-column {
    background: rgba(22, 27, 34, 0.5) !important;
    border: 1px solid var(--nexus-border-color) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(5px);
    padding: 30px !important;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    background-color: var(--nexus-box-color);
    box-shadow: 0 0 15px 2px #00ffff, inset 0 0 10px 1px rgba(0, 255, 255, 0.2);
    border: 2px solid #00ffff !important;

}

.gradio-glow {
    padding: 25px !important;
    box-shadow: 0 0 15px 2px #00ffff, inset 0 0 10px 1px rgba(0, 255, 255, 0.2);
    border: 2px solid #00ffff !important;
    background-color: #111827 !important;
}

/* Indented heading title of section */
#advisor_title h2 {
    font-family: var(--font-title);
    color: var(--nexus-primary-accent);
    border-bottom: 1px solid var(--nexus-border-color);
    padding-bottom: 8px;
    margin-top: 0px !important;
    margin-bottom: 24px !important;
}

/* Gap between resume and skill panels */
#advisor_content_row {
    display: flex;
    gap: 24px;
}

/* Labels above dropdowns, files, etc. */
label.gradio-label > .label-span {
    font-family: var(--font-title) !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    color: var(--nexus-text-color-light) !important;
    background: var(--nexus-label-bg) !important;
    padding: 6px 12px !important;
    border-radius: 8px !important;
    margin-bottom: 8px !important;
    display: inline-block !important;
}

/* Avoids all-caps on tab labels */
.gradio-tabs > .tab-nav > button, .gradio-accordion > .label-wrap > .label-span {
    text-transform: none !important;
}

.gradio-form > div:has(> .gradio-label) {
    margin-bottom: 8px;
}

/* Text and file input styles */
textarea, input[type="text"], .gradio-file, .gradio-image, .gradio-markdown {
    background-color: rgba(13, 17, 23, 0.7) !important;
    border-radius: 12px !important;
    border: 1px solid var(--nexus-border-color) !important;
    color: var(--nexus-text-color) !important;
    box-shadow: inset 0px 1px 4px rgba(0,0,0,0.4);
    padding: 16px !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: var(--nexus-primary-accent) !important;
    box-shadow: 0 0 8px var(--nexus-primary-accent) !important;
}
.gradio-file { border-style: dashed !important; }

/* Dropdowns */
.gradio-dropdown {
    background-color: #161b22 !important;
    border: 1px solid var(--nexus-border-color) !important;
    border-radius: 8px !important;
    color: var(--nexus-text-color-light) !important;
    margin-bottom: 15px;
}

/* Analyze Button */
#analyze_button {
    margin-top: 16px !important;
    margin-bottom: 24px !important;
    border-radius: 10px !important;
    font-family: 'Orbitron', sans-serif !important;  /* modern futuristic style */
    font-weight: 700 !important;
    font-size: 1.1em !important;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    color: #e0e0ff !important;
    background: linear-gradient(90deg, #3b82f6, #6366f1) !important;
    border: none !important;
    transition: all 0.3s ease;
    box-shadow: 0 0 8px #00bfff, 0 0 12px #00bfff;
    text-shadow: 0 0 3px #00bfff, 0 0 6px #00bfff;

}
#analyze_button:hover {
    transform: translateY(-3px);
    color: #000 !important;
    background: var(--nexus-primary-accent) !important;
    box-shadow: 0 0 12px #00ffff, 0 0 24px #00ffff;
}
/* send Button */
#send_button {
    margin-top: 12px !important;
    margin-bottom: 16px !important;
    border-radius: 12px !important;
    font-family: 'Orbitron', sans-serif !important;  /* modern futuristic style */
    font-weight: 700 !important;
    font-size: 1.1em !important;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    color: #e0e0ff !important;
    background: linear-gradient(90deg, #3b82f6, #6366f1) !important;
    border: none !important;
    transition: all 0.3s ease;
    box-shadow: 0 0 8px #00bfff, 0 0 12px #00bfff;
    text-shadow: 0 0 3px #00bfff, 0 0 6px #00bfff;
}

#send_button:hover {
    transform: translateY(-3px);
    color: #000 !important;
    background: #00ffff !important;
    box-shadow: 0 0 12px #00ffff, 0 0 24px #00ffff;
    text-shadow: none;
}


/* ChatBot Glow Container */
.gradio-chatbot-glow {
    display: flex !important;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    box-shadow: 0 0 15px 4px #00ffff, inset 0 0 10px 1px rgba(0, 255, 255, 0.2);
    border: 2px solid #00ffff !important;
    background: var(--nexus-bg-deep-blue) !important;
    min-height: 500px;
    padding: 10px !important;
}
.gradio-chatbot {
    background: none !important;
    border: none !important;
    box-shadow: none !important;
}
.gradio-chatbot > .chatbot-container {
    background: none !important;
    box-shadow: none !important;
}
.welcome-message-container {
    text-align: center;
    pointer-events: none;
}
.welcome-message-text {
    font-family: var(--font-title);
    font-size: 2.8em;
    color: var(--nexus-text-color-light);
    text-shadow: 0 0 10px var(--nexus-secondary-accent);
}
.welcome-message-subtext {
    font-family: var(--font-main);
    font-size: 1.2em;
    color: var(--nexus-text-color);
    margin-top: 10px;
}
#gap_analysis_heading {
    color: #00ffff; /* Neon light blue */
    font-family: var(--font-title);
    font-weight: 700;
    text-shadow: 0 0 6px #00ffff;
}
/* --- [ULTRA-HD] Professional Footer --- */
#app_footer {
    text-align: center !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 1em !important;
    letter-spacing: 2px;
    padding: 40px 20px 35px 20px !important;
    margin-top: 50px;
    position: relative; /* Essential for positioning pseudo-elements */
    border-top: 1px solid rgba(0, 224, 255, 0.2);
    border-bottom: 1px solid rgba(0, 224, 255, 0.2);
    background: linear-gradient(to right, 
        rgba(2, 11, 38, 0.1) 0%, 
        rgba(0, 224, 255, 0.05) 50%, 
        rgba(2, 11, 38, 0.1) 100%);
    box-shadow: 0 -5px 15px -5px rgba(0, 224, 255, 0.3)
}

/* --- Corner Bracket Styling (Pseudo-elements) --- */
#app_footer::before, #app_footer::after {
    content: '';
    position: absolute;
    width: 20px;
    height: 20px;
    transition: all 0.4s ease;
    animation: bracketGlow 3s infinite alternate;
}

#app_footer::before {
    top: 10px;
    left: 10px;
    border-top: 2px solid var(--nexus-primary-accent);
    border-left: 2px solid var(--nexus-primary-accent);
}

#app_footer::after {
    bottom: 10px;
    right: 10px;
    border-bottom: 2px solid var(--nexus-primary-accent);
    border-right: 2px solid var(--nexus-primary-accent);
}

/* --- The Text Itself (Multi-layered Glow) --- */
#app_footer p, #app_footer code {
    font-family: 'Orbitron', sans-serif !important;
    color: #fff !important;
    text-transform: uppercase;
    background: none !important;
    animation: textFlicker 5s infinite alternate;
    /* Layered text-shadow for a deep "bloom" effect */
    text-shadow: 
        0 0 2px #fff,
        0 0 7px var(--nexus-primary-accent),
        0 0 12px var(--nexus-primary-accent),
        0 0 25px var(--nexus-secondary-accent);
}

/* --- KEYFRAME ANIMATIONS for dynamic effects --- */

/* Animation for the flickering/pulsing text */
@keyframes textFlicker {
    0%, 18%, 22%, 25%, 53%, 57%, 100% {
        opacity: 1;
        text-shadow: 
            0 0 2px #fff,
            0 0 7px var(--nexus-primary-accent),
            0 0 12px var(--nexus-primary-accent),
            0 0 25px var(--nexus-secondary-accent);
    }
    20%, 24%, 55% {
        opacity: 0.6;
        text-shadow: none;
    }
}

/* Animation for the glowing corner brackets */
@keyframes bracketGlow {
    from {
        box-shadow: 0 0 5px -2px var(--nexus-primary-accent);
        opacity: 0.6;
    }
    to {
        box-shadow: 0 0 12px 2px rgba(0, 224, 255, 0.5);
        opacity: 1;
    }
}
"""

def clear_textbox():
    return ""

def hide_welcome_message():
    return gr.update(visible=False)

# --- Build the Gradio App ---
with gr.Blocks(theme=gr.themes.Soft(), title="AI Career Suite", css=custom_css) as app:
    gr.Markdown("# 🤖 WELCOME TO THE NEXUS", elem_id="gradio-title")
    session_id_state = gr.State(value=lambda: str(uuid4()))

    with gr.Row():
        mode_selector = gr.Dropdown(
            choices=["1. AI Career Advisor", "2. Smart Academic Mentor"],
            value="1. AI Career Advisor",
            label="Select a Tool from the Suite"
        )
    with gr.Row():
        llm_selector = gr.Dropdown(
            label="Select LLM Model",
            choices=['llama3-70b-8192', 'llama3-8b-8192','meta-llama/llama-4-scout-17b-16e-instruct','meta-llama/llama-4-maverick-17b-128e-instruct','deepseek-r1-distill-llama-70b','qwen/qwen3-32b'],
            value='llama3-70b-8192'
        )

    # --- UI Container for Advisor ---
    with gr.Column(visible=True, elem_classes="gradio-glow") as advisor_ui_container:
        gr.Markdown("## In-Depth Resume & Job Description Analysis", elem_id="advisor_title")
        # --- MODIFIED: Removed 'gap' and added 'elem_id' to control spacing via CSS ---
        with gr.Row(elem_id="advisor_content_row"):
            with gr.Column(scale=1, min_width=350):
                resume_input = gr.File(label="Upload Your Resume (PDF or DOCX)", file_types=[".pdf", ".docx"])
                jd_input = gr.Textbox(lines=15, label="Paste Job Description Here")
                analyze_button = gr.Button("Analyze", variant="primary", elem_id="analyze_button")
            with gr.Column(scale=2, min_width=350):
                gr.Markdown("### Gap Analysis Report",elem_id="gap_analysis_heading")
                chart_output = gr.Image(label="Visual Skill Gap")
                analysis_output = gr.Markdown()
                learning_goal_output = gr.Textbox(label="Suggested Learning Goal (for Mentor)", interactive=False)

    # --- UI Container for Mentor ---
    with gr.Column(visible=False) as mentor_ui_container:
        gr.Markdown("## 🎯 Get a Personalized Learning Plan")
        gr.Markdown("**1️⃣ Start with the goal suggested by the Advisor, or type your own.**")
        gr.Markdown("**2️⃣ You can ask for a 'visual roadmap' or a 'learning chart' to guide your path.**")
        gr.Markdown("**3️⃣ Want a quiz on any topic? Ask Nexus to play the Quiz Game with you!**")
        gr.Markdown("**🔴INSTRUCTION : For use Smart Academic Mentor use models 'meta-llama/llama-4-scout-17b-16e-instruct','meta-llama/llama-4-maverick-17b-128e-instruct'**")
        gr.Markdown("**Don't use other models otherwise you get json text  output !!**")
        with gr.Column(elem_classes="gradio-chatbot-glow") as chatbot_container:
            with gr.Row(visible=True) as welcome_row:
                 welcome_html = gr.HTML(
                    value="""
                    <div class="welcome-message-container">
                        <p class="welcome-message-text">NEXUS</p>
                        <p class="welcome-message-subtext">Hello, I am here to assist you.</p>
                    </div>
                    """
                )
            chatbot = gr.Chatbot(show_label=False)
        # MODIFIED: Input row now includes a voice input button
        with gr.Row():
            voice_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Voice Chat 🎙️",
                scale=1,
            )
            msg = gr.Textbox(
                show_label=False,
                placeholder="Enter your message, record your voice, or press enter",
                scale=6,
            )
            send_button = gr.Button("Send", variant="primary", scale=1, elem_id="send_button")
    gr.Markdown("`[ Nexus Core v1.0 // Engineered by Sweet Poison ]`", elem_id="app_footer")

    # --- EVENT HANDLERS (UNCHANGED) ---
    analyze_button.click(
        fn=run_career_advisor_analysis,
        inputs=[resume_input, jd_input, llm_selector],
        outputs=[analysis_output, learning_goal_output, chart_output]
    )

    send_button.click(
        fn=mentor_chat_response,
        inputs=[msg, chatbot, session_id_state, llm_selector],
        outputs=[chatbot]
    ).then(
        fn=lambda: gr.update(visible=False),
        outputs=[welcome_row]
    ).then(
        fn=clear_textbox,
        outputs=[msg]
    )

    msg.submit(
        fn=mentor_chat_response,
        inputs=[msg, chatbot, session_id_state, llm_selector],
        outputs=[chatbot]
    ).then(
        fn=lambda: gr.update(visible=False),
        outputs=[welcome_row]
    ).then(
        fn=clear_textbox,
        outputs=[msg]
    )
    
    # NEW: Event handler for voice input. It transcribes audio and then calls the original chat function.
    voice_input.stop_recording(
        fn=transcribe_audio,
        inputs=[voice_input],
        outputs=[msg]
    ).then(
        fn=mentor_chat_response,
        inputs=[msg, chatbot, session_id_state, llm_selector],
        outputs=[chatbot]
    ).then(
        fn=lambda: gr.update(visible=False),
        outputs=[welcome_row]
    ).then(
        fn=clear_textbox,
        outputs=[msg]
    )

    # Hide welcome message on any input activity (text or voice)
    msg.change(fn=hide_welcome_message, outputs=[welcome_row])
    voice_input.start_recording(fn=hide_welcome_message, outputs=[welcome_row])
    
    mode_selector.change(
        fn=switch_ui_mode,
        inputs=mode_selector,
        outputs=[advisor_ui_container, mentor_ui_container]
    )

# --- The rest of your script (if __name__ == "__main__":) ---
if __name__ == "__main__":
    if "GROQ_API_KEY" not in os.environ or not os.environ["GROQ_API_KEY"]:
        print("Cannot launch app: GROQ_API_KEY is not set in your environment or .env file.")
    else:
        if not os.path.exists("bot.png"):
            try:
                img = Image.new('RGB', (100, 100), color = 'darkgray')
                img.save('bot.png')
                print("Created dummy 'bot.png' avatar.")
            except Exception as e:
                print(f"Could not create dummy 'bot.png': {e}. Please add a 'bot.png' file to the directory.")
        print("Launching Gradio App... Open the URL in your browser.")
        app.launch(share=True)
        
        