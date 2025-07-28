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

# --- 1. SET UP THE ENVIRONMENT & LLM (WITH GROQ) ---
load_dotenv()

if "GROQ_API_KEY" not in os.environ:
    print("Error: GROQ_API_KEY not found in .env file.")
    llm = None
else:
    try:
        llm = ChatGroq(
            temperature=0.1,
            model_name="llama3-70b-8192"
        )
    except Exception as e:
        print(f"Error initializing the LLM: {e}")
        llm = None

# --- 2. CHART GENERATION FUNCTIONS ---
# --- UNCHANGED CAREER ADVISOR CHART ---
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

# --- Make sure these imports are at the top of your Python file ---
import textwrap
from matplotlib.font_manager import FontProperties
import matplotlib.patheffects as path_effects

def create_learning_roadmap_chart(roadmap_data):
    """
    Generates and saves a high-quality, professional Gantt-style chart for a learning roadmap.

    This definitive version focuses on clarity and aesthetics by:
    1.  Dramatically increasing vertical spacing to eliminate all text overlap.
    2.  Intelligently sizing bars based on both duration and text length to prevent clipping.
    3.  Wrapping long milestone descriptions to fit neatly under their respective topics.
    4.  Fine-tuning fonts, spacing, and resolution for a polished, top-notch visual.
    """
    if not roadmap_data or 'roadmap' not in roadmap_data:
        return None

    # --- Data Extraction ---
    topics = [item['topic'] for item in roadmap_data['roadmap']]
    durations = [item['duration_weeks'] for item in roadmap_data['roadmap']]
    stages = [item['stage'] for item in roadmap_data['roadmap']]
    milestones = [item.get('milestone') for item in roadmap_data['roadmap']]

    # --- Color and Stage Setup ---
    unique_stages = sorted(list(set(stages)), key=stages.index)
    colors = plt.colormaps.get_cmap('viridis')(np.linspace(0, 1, len(unique_stages)))
    stage_colors = {stage: colors[i] for i, stage in enumerate(unique_stages)}

    # --- KEY CHANGE: Increase figure size for more vertical spacing ---
    fig, ax = plt.subplots(figsize=(25, len(topics) * 1.5 + 3))
    y_pos = np.arange(len(topics))
    start_time = 0

    # --- Font Handling for Emoji ---
    try:
        emoji_font = FontProperties(family=['Segoe UI Emoji', 'Arial'])
    except:
        emoji_font = FontProperties(family=['Arial']) # Fallback font

    for i in range(len(topics)):
        bar_color = stage_colors[stages[i]]

        # --- KEY CHANGE: Smart bar width calculation ---
        # Make bar width responsive to text length to avoid cramping.
        topic_len_factor = len(topics[i]) / 15.0
        min_width = 3.5 + topic_len_factor
        display_duration = max(durations[i], min_width)
        
        # --- Draw the bar with more thickness ---
        ax.barh(y_pos[i], display_duration, left=start_time, align='center',
                      height=0.6, color=bar_color, edgecolor='black', linewidth=1.5)

        # --- Dynamic and adaptive text inside the bar ---
        r, g, b, _ = bar_color
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = 'white' if luminance < 0.5 else 'black'
        
        wrap_width = max(int(display_duration * 2.8), 15) # Ensure wrap width is reasonable
        wrapped_topic = textwrap.fill(topics[i], width=wrap_width)
        
        ax.text(start_time + display_duration/2, y_pos[i], wrapped_topic,
                va='center', ha='center', color=text_color, fontweight='bold', fontsize=10.5)

        # --- KEY CHANGE: Position and wrap milestone text to prevent overlap ---
        if milestones[i] and milestones[i].strip():
            # Wrap long milestone text
            wrapped_milestone = textwrap.fill(f"⭐ {milestones[i]}", width=70)
            
            # Position the text below the bar with sufficient padding
            milestone_text = ax.text(start_time, y_pos[i] + 0.45, wrapped_milestone,
                                     va='top', ha='left', color='gold',
                                     fontsize=16, fontweight='bold', fontproperties=emoji_font)
            
            # Add a strong outline for readability
            milestone_text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'),
                                             path_effects.Normal()])

        start_time += durations[i]

    # --- Chart Styling and Aesthetics ---
    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel('Timeline in Weeks', fontsize=15, fontweight='bold')
    ax.set_title(roadmap_data.get('chart_title', 'Learning Roadmap'), fontsize=22, fontweight='bold', pad=35)
    
    # --- Legend ---
    patches = [mpatches.Patch(color=color, label=stage) for stage, color in stage_colors.items()]
    legend = ax.legend(handles=patches, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
    legend.get_frame().set_edgecolor('black')

    # --- Final Touches ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    ax.set_xlim(0, start_time + 2) # Add padding to the end of the timeline

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    
    filepath = f"roadmap_chart_{uuid4()}.png"
    plt.savefig(filepath, dpi=200, bbox_inches='tight') # Higher DPI for crispness
    plt.close(fig)
    
    return filepath

# --- 3. HELPER FUNCTIONS (UNCHANGED) ---
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
        output_str += f"> `{', '.join(keywords_matched) if keywords_matched else 'None found.'}`\n\n"
        output_str += "**Critical keywords missing from your resume:**\n"
        output_str += f"> `{', '.join(keywords_missing) if keywords_missing else 'None identified.'}`\n\n---\n\n"
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

# --- 4. MODULE 1: AI CAREER ADVISOR LOGIC (VERIFIED AS UNCHANGED) ---
def run_career_advisor_analysis(resume_file, job_description):
    if llm is None: return "LLM not initialized. Check your GROQ_API_KEY.", "", None
    if not resume_file or not job_description: return "Please upload a resume and provide a job description.", "", None

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
        
        print("\n--- DEBUG: RAW RESPONSE FROM LLM ---\n", raw_response, "\n--- END RAW RESPONSE ---\n")
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

# --- 5. MODULE 2: SMART ACADEMIC MENTOR LOGIC (UNCHANGED) ---
session_memories = {}

def get_mentor_chain(session_id):
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    mentor_prompt_template = """You are a helpful and friendly Smart Academic Mentor. Your primary goal is to help users break down their learning goals into a structured, actionable plan.
    **SPECIAL INSTRUCTIONS FOR VISUALIZATION:**
    1.  Analyze the user's request. If the user explicitly asks for a "chart," "roadmap," "visual," "graph," or "timeline" for their learning plan, you MUST generate a special JSON object for visualization.
    2.  This JSON object must be placed at the very end of your response, enclosed in ```json ... ``` tags.
    3.  The JSON object must have this exact structure (NOTE: The curly braces {{ and }} are escaped for the template):
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
    4.  Even when generating the JSON, you must ALSO provide your normal, conversational, and encouraging step-by-step learning plan in Markdown format BEFORE the JSON block. The JSON is for the system; the text is for the user.
    5. If the user is just chatting or asks a question that is NOT a request for a full learning roadmap visual in clear vision and all writing of the chart must be clear, DO NOT generate the JSON. Just respond conversationally. Also, ensure that when generating the chart:The chart is highly visible, with no overlapping between numbers, text, boxes, or lines.It should be designed in such a way that users don't have to zoom in to read any part of the chart.And the output must be crisp, non-blurry, and of premium quality, ensuring maximum readability and professional appearance.
    Previous conversation:
    {chat_history}
    New question: {input}
    Your Response:
    """
    mentor_prompt = PromptTemplate.from_template(mentor_prompt_template)
    return LLMChain(llm=llm, prompt=mentor_prompt, memory=session_memories[session_id], verbose=True)

# UNCHANGED - This function correctly manages history for a manual chatbot UI
def mentor_chat_response(message, history, session_id):
    if llm is None:
        history.append([message, "Error: LLM not initialized. Check your GROQ_API_KEY."])
        return history

    # Append the user's message to history immediately
    history.append([message, None])
    
    chain = get_mentor_chain(session_id)
    raw_response = chain.predict(input=message)

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
    
    # Update the last history entry (which was the user message) with the bot's response
    history[-1][1] = text_response
    
    # If a chart was created, add a new entry to the history for the image
    if chart_path:
        history.append([None, (chart_path,)])
        
    return history


# --- 6. GRADIO UI AND APPLICATION LAUNCH (MODIFIED WITH DROPDOWN) ---

def switch_ui_mode(mode_selection):
    """
    Controller function to switch visibility of UI sections based on dropdown value.
    Returns a dictionary of Gradio updates.
    """
    is_advisor_visible = (mode_selection == "1. AI Career Advisor")
    is_mentor_visible = (mode_selection == "2. Smart Academic Mentor")
    
    return {
        advisor_ui_container: gr.update(visible=is_advisor_visible),
        mentor_ui_container: gr.update(visible=is_mentor_visible)
    }

with gr.Blocks(theme=gr.themes.Soft(), title="AI Career Suite") as app:
    gr.Markdown("# 🤖 GenAI Career Suite (Powered by Groq)")
    session_id_state = gr.State(value=lambda: str(uuid4()))

    # --- NEW: Dropdown selector for choosing the tool ---
    mode_selector = gr.Dropdown(
        choices=["1. AI Career Advisor", "2. Smart Academic Mentor"],
        value="1. AI Career Advisor",
        label="Select a Tool from the Suite"
    )

    # --- Container for the "AI Career Advisor" UI ---
    with gr.Column(visible=True) as advisor_ui_container:
        gr.Markdown("## In-Depth Resume & Job Description Analysis")
        with gr.Row():
            with gr.Column(scale=1):
                resume_input = gr.File(label="Upload Your Resume (PDF or DOCX)", file_types=[".pdf", ".docx"])
                jd_input = gr.Textbox(lines=15, label="Paste Job Description Here")
                analyze_button = gr.Button("Analyze", variant="primary")
            with gr.Column(scale=2):
                gr.Markdown("### Gap Analysis Report")
                chart_output = gr.Image(label="Visual Skill Gap")
                analysis_output = gr.Markdown()
                learning_goal_output = gr.Textbox(label="Suggested Learning Goal (for Mentor)", interactive=False)

    # --- Container for the "Smart Academic Mentor" UI ---
    with gr.Column(visible=False) as mentor_ui_container:
        gr.Markdown("## Get a Personalized Learning Plan")
        gr.Markdown("Start with the goal from the Advisor, or type your own. **Try asking for a 'visual roadmap' or a 'learning chart'!**")
        
        chatbot = gr.Chatbot(height=500, show_label=False)
        
        with gr.Row():
            msg = gr.Textbox(
                show_label=False,
                placeholder="Enter your message and press enter, or click Send",
                scale=7,
            )
            send_button = gr.Button("Send", variant="primary", scale=1)

        # Function to clear the textbox after message is sent
        def clear_textbox():
            return ""

        # Event handler for the "Send" button
        send_button.click(
            fn=mentor_chat_response,
            inputs=[msg, chatbot, session_id_state],
            outputs=[chatbot]
        ).then(fn=clear_textbox, outputs=[msg])

        # Event handler for pressing Enter in the textbox
        msg.submit(
            fn=mentor_chat_response,
            inputs=[msg, chatbot, session_id_state],
            outputs=[chatbot]
        ).then(fn=clear_textbox, outputs=[msg])

    # --- Connect dropdown changes to the UI switching function ---
    mode_selector.change(
        fn=switch_ui_mode,
        inputs=mode_selector,
        outputs=[advisor_ui_container, mentor_ui_container]
    )
            
    # Connect the analyze button to its function (remains unchanged)
    analyze_button.click(
        fn=run_career_advisor_analysis,
        inputs=[resume_input, jd_input],
        outputs=[analysis_output, learning_goal_output, chart_output]
    )

if __name__ == "__main__":
    if llm is None:
        print("Cannot launch app because LLM initialization failed. Ensure your GROQ_API_KEY is set in your environment.")
    else:
        if not os.path.exists("bot.png"):
            try:
                img = Image.new('RGB', (100, 100), color = 'darkgray')
                img.save('bot.png')
                print("Created dummy 'bot.png' avatar.")
            except Exception as e:
                print(f"Could not create dummy 'bot.png': {e}. Please add a 'bot.png' file to the directory.")
        print("Launching Gradio App... Open the URL in your browser.")
        app.launch()