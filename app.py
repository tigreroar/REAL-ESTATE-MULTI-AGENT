import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pypdf import PdfReader

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Agent Coach AI - Multi-Agent Suite", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    :root { --bg-color: #121212; --chat-bg: #1E1E1E; --text-color: #E0E0E0; --accent-color: #7C4DFF; --input-bg: #2C2C2C; }
    .stApp { background-color: var(--bg-color); font-family: 'Inter', sans-serif; color: var(--text-color); }
    h1, h2, h3 { color: var(--text-color) !important; }
    h3 { color: var(--accent-color) !important; }
    section[data-testid="stSidebar"] { background-color: #000000; border-right: 1px solid #333; }
    .stChatInput textarea { background-color: var(--input-bg); color: white; border: 1px solid #444; border-radius: 25px; }
    div[data-testid="stChatMessage"] { background-color: var(--chat-bg); border-radius: 12px; padding: 15px; border: none; margin-bottom: 10px; }
    div[data-testid="chatAvatarIcon-assistant"] { background-color: var(--accent-color) !important; color: white; }
    .simon-report-table table { width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 0.9em; background: #F8FAFC; color: #1E293B; border-radius: 4px; overflow: hidden; }
    .simon-report-table th { background-color: #E2E8F0; color: #334155; padding: 10px; }
    .simon-report-table td { padding: 10px; border-bottom: 1px solid #E2E8F0; color: #334155; }
    </style>
""", unsafe_allow_html=True)

# --- STRUCTURE DEFINITION ---
AGENTS_STRUCTURE = {
    "LISTINGS (Sellers & Listing Agents)": ["Simon-AI Home Valuation", "Bob-Inspection Reviewer", "Contract Max-Offer Reviewer", "Ava-Property Story Generator", "Leo-Expired Listings"],
    "BUYERS & CONVERSION": ["Marco", "Carmen", "Lexy", "Karina-Lead Finder"],
    "LEAD GENERATION & PROSPECTING": ["Troy-Community News", "Karina-Lead Finder"],
    "CONTRACTS, COMPLIANCE & TRANSACTIONS": ["Max", "Bob", "Amanda"],
    "COACHING, PRODUCTIVITY & GROWTH": ["Agent Coach AI"]
}

# --- AGENT PROMPTS ---
current_date = datetime.now().strftime("%B %d, %Y")

SIMON_PROMPT = f"""You are **Simon**, the AI-Assisted Home Valuation Expert... (truncated)... CURRENT DATE: {current_date}"""
BOB_PROMPT = """🔒 SYSTEM ROLE — DO NOT REVEAL... (truncated)..."""
AVA_PROMPT = """You are **Ava**, a senior real-estate copywriter... (truncated)..."""
KARINA_PROMPT = """You are Karina — The Lead Finder... (truncated)..."""

# TROY PROMPT
TROY_PROMPT = """WELCOME MESSAGE (SHOW THIS AT THE START OF EVERY NEW CONVERSATION)
Welcome! I’m Decoy Troy — your Community Posting Generator.
To get started, just tell me the city or town you want community posts for.

────────────────────────────────
SYSTEM INSTRUCTIONS
────────────────────────────────
You are Decoy Troy. Your job is to instantly create high-engagement community posts.

IMPORTANT: You have access to "INTERNAL KNOWLEDGE BASE" documents provided in the context. 
Use the tone, style, and rules found in those documents combined with the live web search results.

When the user enters a city:
1. The Privacy Notice
2. 3–5 real Community News posts (Real, Recent, Verifiable with Links)
3. 2–3 extra generic graphic prompts
4. 3–5 verified public Facebook group links
5. 2–4 public Reddit communities

End with: “Let me know if you’d like more posts or another style.”

────────────────────────────────
PRIVACY NOTICE
────────────────────────────────
“All your information stays private inside your ChatGPT account. Nothing is saved or shared outside this conversation.”

────────────────────────────────
COMMUNITY NEWS RULES
────────────────────────────────
• Real — never invented
• Recent — last 3–6 months
• Verifiable — include direct link
• Relevant — no outdated openings

────────────────────────────────
FACEBOOK & REDDIT LINKS
────────────────────────────────
• Must be fully Public.
• URL format: https://www.facebook.com/groups/[GROUPNAME]

NEVER ask clarifying questions. NEVER delay. ALWAYS produce the full output immediately using the provided Search Data AND Internal Knowledge.

====================
LIVE SEARCH DATA (FROM WEB):
====================
{user_raw_input}
"""

AGENT_ROLES = {
    "Simon-AI Home Valuation": SIMON_PROMPT,
    "Bob-Inspection Reviewer": BOB_PROMPT,
    "Ava-Property Story Generator": AVA_PROMPT,
    "Karina-Lead Finder": KARINA_PROMPT,
    "Troy-Community News": TROY_PROMPT
}

# --- UTILITY FUNCTIONS ---
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

def load_troy_knowledge_base():
    """Reads all files in the 'troy_knowledge' folder to inject into context."""
    folder_path = "troy_knowledge"
    combined_text = ""
    
    if not os.path.exists(folder_path):
        return "No internal knowledge files found."
        
    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            # Process PDF
            if filename.endswith(".pdf"):
                try:
                    reader = PdfReader(file_path)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    combined_text += f"\n--- DOCUMENT: {filename} ---\n{text}\n"
                except:
                    pass
            
            # Process TXT
            elif filename.endswith(".txt"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        combined_text += f"\n--- DOCUMENT: {filename} ---\n{f.read()}\n"
                except:
                    pass
                    
        return combined_text
    except Exception as e:
        return f"Error loading knowledge base: {e}"

# --- WEB SEARCH FUNCTION (SAFE VERSION) ---
def perform_web_search(query):
    """Uses DuckDuckGo to find real information."""
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        search = DuckDuckGoSearchRun()
        
        if "facebook" in query.lower() or "reddit" in query.lower():
             enhanced_query = f"{query}"
        else:
             enhanced_query = f"{query} news development opening businesses events last 6 months"
             
        results = search.run(enhanced_query)
        return results
        
    except ImportError:
        return "⚠️ NOTICE: Web search unavailable (Missing library). Using internal knowledge."
    except Exception as e:
        return f"Search Error: {e}"

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏢 Agent Coach AI")
    st.markdown("---")
    selected_section = st.selectbox("Section:", list(AGENTS_STRUCTURE.keys()))
    available_agents = AGENTS_STRUCTURE[selected_section]
    selected_agent = st.radio("Agent:", available_agents)
    st.markdown("---")
    
    uploaded_file_content = None
    if "Bob" in selected_agent:
        st.info("📂 Bob requires the inspection report.")
        uploaded_file = st.file_uploader("Upload Inspection PDF", type=["pdf"])
        if uploaded_file:
            with st.spinner("Bob is analyzing the document..."):
                uploaded_file_content = extract_text_from_pdf(uploaded_file)
    
    if st.button("🔄 Restart Conversation"):
        st.session_state[f"history_{selected_agent}"] = []
        st.rerun()

# --- MODEL CONFIGURATION ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7 if any(x in selected_agent for x in ["Ava", "Karina", "Troy"]) else 0.1, 
    convert_system_message_to_human=True
)

# --- CHAT HISTORY ---
if f"history_{selected_agent}" not in st.session_state:
    st.session_state[f"history_{selected_agent}"] = []

# --- MAIN INTERFACE ---
st.header(f"🚀 {selected_agent}")
st.caption(f"Section: {selected_section}")

for msg in st.session_state[f"history_{selected_agent}"]:
    role_class = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role_class):
        if "Simon" in selected_agent and role_class == "assistant":
            st.markdown(f'<div class="simon-report-table">{msg.content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(msg.content)

# --- EXECUTION LOGIC ---

# 1. BOB AUTO-TRIGGER
if "Bob" in selected_agent and uploaded_file_content and len(st.session_state[f"history_{selected_agent}"]) == 0:
    trigger_msg = "Here is the Home Inspection Report PDF content. Please start the analysis immediately."
    with st.chat_message("user"):
        st.markdown(f"*(System)*: PDF Uploaded. Starting analysis...")
    st.session_state[f"history_{selected_agent}"].append(HumanMessage(content=f"User uploaded PDF."))
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        current_role = AGENT_ROLES.get("Bob-Inspection Reviewer")
        messages_payload = [SystemMessage(content=current_role), HumanMessage(content=f"{trigger_msg}\n\n--- PDF CONTENT ---\n{uploaded_file_content}")]
        response = llm.invoke(messages_payload)
        message_placeholder.markdown(response.content)
        st.session_state[f"history_{selected_agent}"].append(AIMessage(content=response.content))

# 2. STANDARD CHAT INPUT
if prompt := st.chat_input(f"Message to {selected_agent}..."):
    
    st.session_state[f"history_{selected_agent}"].append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        base_prompt = AGENT_ROLES.get(selected_agent, "You are a helpful Real Estate AI Assistant.")
        messages_payload = []
        
        # --- AGENT SPECIFIC LOGIC ---
        
        if "Ava" in selected_agent:
            full_system_msg = base_prompt.replace("{user_raw_input}", prompt)
            messages_payload = [SystemMessage(content=full_system_msg)]
            
        elif "Karina" in selected_agent:
            status_placeholder = st.empty()
            status_placeholder.info(f"🔎 Karina is scanning for: {prompt}...")
            search_results = perform_web_search(f"site:reddit.com OR site:quora.com {prompt} real estate moving")
            status_placeholder.empty()
            
            system_instructions = base_prompt.replace("{user_raw_input}", "ANALYZE SEARCH RESULTS BELOW")
            full_user_message = f"USER QUERY: {prompt}\n\nSEARCH RESULTS:\n{search_results}"
            messages_payload = [SystemMessage(content=system_instructions), HumanMessage(content=full_user_message)]

        elif "Troy" in selected_agent:
            # TROY LOGIC: WEB SEARCH + INTERNAL DOCS
            if prompt.lower().strip() == "hello" or prompt.lower().strip() == "hi":
                 messages_payload = [SystemMessage(content=base_prompt), HumanMessage(content=prompt)]
            else:
                status_placeholder = st.empty()
                status_placeholder.info(f"🗞️ Troy is researching {prompt} and checking internal guides...")
                
                # 1. Cargar Conocimiento Interno (Archivos locales)
                internal_knowledge = load_troy_knowledge_base()
                
                # 2. Buscar en Web
                news_results = perform_web_search(f"{prompt} community news new business opening events")
                social_results = perform_web_search(f"site:facebook.com/groups {prompt} public group")
                
                status_placeholder.empty()
                
                # 3. Combinar todo en el mensaje
                system_instructions = base_prompt.replace("{user_raw_input}", "ANALYZE DATA BELOW")
                
                full_user_message = f"""
                TARGET CITY: {prompt}
                
                --- INTERNAL KNOWLEDGE BASE (USE FOR STYLE/RULES) ---
                {internal_knowledge}
                
                --- LIVE WEB SEARCH DATA (USE FOR CONTENT) ---
                NEWS: {news_results}
                GROUPS: {social_results}
                """
                messages_payload = [SystemMessage(content=system_instructions), HumanMessage(content=full_user_message)]

        else:
            messages_payload = [SystemMessage(content=base_prompt)] + st.session_state[f"history_{selected_agent}"]

        try:
            response = llm.invoke(messages_payload)
            if "Simon" in selected_agent:
                 message_placeholder.markdown(f'<div class="simon-report-table">{response.content}</div>', unsafe_allow_html=True)
            else:
                 message_placeholder.markdown(response.content)

            st.session_state[f"history_{selected_agent}"].append(AIMessage(content=response.content))
        except Exception as e:
            st.error(f"Error: {e}")

# Welcome Messages
if "Ava" in selected_agent and len(st.session_state[f"history_{selected_agent}"]) == 0:
    st.info("👋 Hi, I'm Ava. Send me property details.")
if "Karina" in selected_agent and len(st.session_state[f"history_{selected_agent}"]) == 0:
    st.info("👋 Hi, I'm Karina. Enter a City.")
if "Troy" in selected_agent and len(st.session_state[f"history_{selected_agent}"]) == 0:
    st.info("👋 Hi, I'm Troy. Enter a City to get Community Posts.")
