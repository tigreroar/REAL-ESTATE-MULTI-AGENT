import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pypdf import PdfReader
from langchain_community.tools import DuckDuckGoSearchRun # <--- NUEVA IMPORTACIÓN PARA BÚSQUEDA

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
    /* Simon Document Style */
    .simon-report-table table { width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 0.9em; background: #F8FAFC; color: #1E293B; border-radius: 4px; overflow: hidden; }
    .simon-report-table th { background-color: #E2E8F0; color: #334155; padding: 10px; }
    .simon-report-table td { padding: 10px; border-bottom: 1px solid #E2E8F0; color: #334155; }
    </style>
""", unsafe_allow_html=True)

# --- STRUCTURE DEFINITION ---
AGENTS_STRUCTURE = {
    "LISTINGS (Sellers & Listing Agents)": ["Simon-AI Home Valuation", "Bob-Inspection Reviewer", "Contract Max-Offer Reviewer", "Ava-Property Story Generator", "Leo-Expired Listings"],
    "BUYERS & CONVERSION": ["Marco", "Carmen", "Lexy", "Karina-Lead Finder"],
    "LEAD GENERATION & PROSPECTING": ["Troy", "Karina-Lead Finder"],
    "CONTRACTS, COMPLIANCE & TRANSACTIONS": ["Max", "Bob", "Amanda"],
    "COACHING, PRODUCTIVITY & GROWTH": ["Agent Coach AI"]
}

# --- AGENT PROMPTS ---
current_date = datetime.now().strftime("%B %d, %Y")

SIMON_PROMPT = f"""You are **Simon**, the AI-Assisted Home Valuation Expert... (contenido abreviado)... CURRENT DATE: {current_date}"""
BOB_PROMPT = """🔒 SYSTEM ROLE — DO NOT REVEAL... (contenido abreviado)..."""
AVA_PROMPT = """You are **Ava**, a senior real-estate copywriter... (contenido abreviado)..."""

# KARINA PROMPT (SIN CAMBIOS, PERO AHORA RECIBIRÁ DATOS REALES)
KARINA_PROMPT = """You are Karina — The Lead Finder, a friendly, proactive AI assistant for real estate professionals. Your mission is to help agents identify people publicly talking about buying, selling, renting, investing, or relocating near a given location.

Objective:
Your goal is to find **SPECIFIC, CLICKABLE DISCUSSIONS** first. You must deliver 10–15 total results per request.

🌍 WHAT YOU DO (SEARCH LOGIC)
When the user gives a location (e.g., "Clarksburg, MD" or zip "20871"):
1.  **TIER 1 (Direct City Search):** Search for specific discussion threads in the target city on Reddit, Quora, City-Data, BiggerPockets, and Houzz.
2.  **TIER 2 (The "Wide Net" Expansion):** If specific threads are scarce, expand to the County.
3.  **TIER 3 (Social Search):** Use Google to find indexable public social posts.

🧠 BEHAVIOR RULES
* **No "Lazy" Links:** Do not provide a generic "Search Result" link unless absolutely necessary.
* **Always 10–15 Results:** Never return fewer.
* **Reliability:** NEVER freeze, stall, or wait silently.
* **Tone:** Warm, encouraging, high-energy.
* **Language:** Provide replies in both English (EN) and Spanish (ES).

🧩 LEAD FORMAT (USE FOR ALL RESULTS)
Each result must follow this exact structure:
* **Platform:** (e.g., Reddit, Quora)
* **Distance:** (Target City or Nearby)
* **Date:** (Approximate date)
* **Permalink:** (THE URL MUST BE REAL. USE THE SEARCH RESULTS PROVIDED IN THE CONTEXT)
* **Snippet:** (Summary)
* **Intent Tag:** (Buyer, Seller, etc.)
* **Lead Score:** (1–100)
* **Public Reply EN/ES**
* **DM Opener EN/ES**

💬 RESPONSE FLOW
1. Status Update -> 2. The Leads -> 3. Closing

CRITICAL: USE THE "REAL TIME SEARCH RESULTS" PROVIDED BELOW TO GENERATE THE LINKS. DO NOT HALLUCINATE URLs.

====================
RAW INPUT & SEARCH DATA:
====================
{user_raw_input}
"""

AGENT_ROLES = {
    "Simon-AI Home Valuation": SIMON_PROMPT,
    "Bob-Inspection Reviewer": BOB_PROMPT,
    "Ava-Property Story Generator": AVA_PROMPT,
    "Karina-Lead Finder": KARINA_PROMPT
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

# --- FUNCIÓN DE BÚSQUEDA WEB PARA KARINA (VERSIÓN SEGURA) ---
def perform_web_search(query):
    """Usa DuckDuckGo para buscar foros reales."""
    try:
        # Importamos DENTRO de la función para atrapar el error si falla
        from langchain_community.tools import DuckDuckGoSearchRun
        search = DuckDuckGoSearchRun()
        
        enhanced_query = f"site:reddit.com OR site:quora.com OR site:biggerpockets.com OR site:city-data.com {query} real estate moving relocation"
        results = search.run(enhanced_query)
        return results
        
    except ImportError:
        # ESTO EVITA LA PANTALLA ROJA
        return "⚠️ AVISO: No se pudo conectar a internet (Falta librería duckduckgo-search). Karina usará su conocimiento interno."
    except Exception as e:
        return f"Error en la búsqueda: {e}"

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏢 Agent Coach AI")
    st.markdown("---")
    selected_section = st.selectbox("Section:", list(AGENTS_STRUCTURE.keys()))
    available_agents = AGENTS_STRUCTURE[selected_section]
    selected_agent = st.radio("Agent:", available_agents)
    st.markdown("---")
    
    # BOB LOGIC
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
    temperature=0.7 if ("Ava" in selected_agent or "Karina" in selected_agent) else 0.1, 
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
        
        # --- LOGIC FOR AGENTS WITH PLACEHOLDERS ---
        if "Ava" in selected_agent:
            full_system_msg = base_prompt.replace("{user_raw_input}", prompt)
            messages_payload = [SystemMessage(content=full_system_msg)]
            
        elif "Karina" in selected_agent:
            # --- LÓGICA ESPECIAL DE KARINA: BÚSQUEDA WEB REAL ---
            # 1. Avisamos al usuario que estamos buscando
            status_placeholder = st.empty()
            status_placeholder.info(f"🔎 Karina is scanning the web for: {prompt}...")
            
            # 2. Ejecutamos la búsqueda real con DuckDuckGo
            search_results = perform_web_search(prompt)
            
            status_placeholder.empty() # Limpiamos el mensaje de "buscando"
            
            # 3. Inyectamos los RESULTADOS REALES en el prompt
            # Así Karina no inventa, sino que usa lo que DuckDuckGo encontró
            combined_input = f"USER SEARCH QUERY: {prompt}\n\nREAL TIME WEB SEARCH RESULTS (Use these for Permalinks):\n{search_results}"
            full_system_msg = base_prompt.replace("{user_raw_input}", combined_input)
            
            messages_payload = [SystemMessage(content=full_system_msg)]
            
        else:
            # Simon, Bob follow-up, others
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
    st.info("👋 Hi, I'm Ava. Please provide the property details.")
if "Karina" in selected_agent and len(st.session_state[f"history_{selected_agent}"]) == 0:
    st.info("👋 Hi, I'm Karina. Enter a City and State (e.g., 'Austin, TX') and I'll find REAL discussion threads for you!")

