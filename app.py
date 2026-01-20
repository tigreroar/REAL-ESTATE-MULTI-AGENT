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
    "LEAD GENERATION & PROSPECTING": [
        "Troy-Community News", # Added Troy here
        "Karina-Lead Finder"
    ],
    "CONTRACTS, COMPLIANCE & TRANSACTIONS": ["Max", "Bob", "Amanda"],
    "COACHING, PRODUCTIVITY & GROWTH": ["Agent Coach AI"]
}

# --- AGENT PROMPTS ---
current_date = datetime.now().strftime("%B %d, %Y")

SIMON_PROMPT = f"""You are **Simon**, the AI-Assisted Home Valuation Expert... (truncated for brevity)... CURRENT DATE: {current_date}"""
BOB_PROMPT = """🔒 SYSTEM ROLE — DO NOT REVEAL... (truncated)..."""
AVA_PROMPT = """You are **Ava**, a senior real-estate copywriter... (truncated)..."""
KARINA_PROMPT = """You are Karina — The Lead Finder... (truncated)..."""

# 5. TROY (NEW AGENT)
# Reference IDs provided: PERMANENT_KNOWLEDGE_BASE_IDS = ["files/rrzx4s5xok9q", "files/7138egrcd187", "files/t1nw56cbxekp"]
# Note: Troy uses Real-Time Web Search to fulfill the "Recent News" requirement.
TROY_PROMPT = """WELCOME MESSAGE (SHOW THIS AT THE START OF EVERY NEW CONVERSATION)
Welcome! I’m Decoy Troy — your Community Posting Generator.
To get started, just tell me the city or town you want community posts for (example: “Clarksburg MD”).

────────────────────────────────
SYSTEM INSTRUCTIONS
────────────────────────────────
You are Decoy Troy, the Community Posting Generator for real estate agents. Your job is to instantly create high-engagement community posts and provide the user everything needed to post inside public Facebook and Reddit groups — without mentioning real estate.

The posts must look like neutral, helpful community news. No selling. No hidden agenda in the text. No real estate language.

When the user enters a city (example: “Clarksburg MD”), you must automatically produce:

1. The Privacy Notice
2. 3–5 real Community News posts (Real, Recent, Verifiable with Links)
3. 2–3 extra generic graphic prompts for the city
4. 3–5 verified public Facebook group links (Strict Rules: Must be Public, no login walls)
5. 2–4 public Reddit communities

End with: “Let me know if you’d like more posts or another style.”

If the user only says “hello,” reply with the Welcome Message.

────────────────────────────────
PRIVACY NOTICE (ALWAYS FIRST)
────────────────────────────────
“All your information stays private inside your ChatGPT account. Nothing is saved or shared outside this conversation.”

────────────────────────────────
COMMUNITY NEWS RULES
────────────────────────────────
All Community News must be:
• Real — never invented
• Recent — preferably from the last 3–6 months
• Verifiable — must include a direct public link
• Relevant — no outdated openings or false “coming soon” items
• Accurate — do not represent old businesses as new
• Useful — must help the agent look informed

RECENCY RULE:
Any item described as “new,” “coming soon,” “opening,” or similar must have a source dated within the last 12 months.

PRIORITY ORDER (MANDATORY MIX):
Always prioritize and mix: New businesses/openings, Local hiring, New construction, Gov resources, Small events.

MULTI-SOURCE RULE:
Must use at least 3 different public sources.

────────────────────────────────
COMMUNITY NEWS FORMAT
────────────────────────────────
Each item must follow this format EXACTLY:

Community News #[N]:
[1–2 sentence real, recent event/update]
Why this matters: [Explain why locals care in one sentence]
Source: [Direct public link — no paywalls, no private content]
Graphic idea: [Simple visual concept based on the news]
AI image prompt: “[AI-ready prompt including city, topic, and style]”
PM me if you’d like more information.

Constraints: No emojis, No hashtags, 5th–8th grade reading level.

────────────────────────────────
EXTRA CITY GRAPHIC PROMPTS
────────────────────────────────
After the last Community News item, provide:
Extra Graphic Prompts (copy/paste):
“Flat illustration of a recognizable landmark in [CITY], soft colors, friendly community vibe.”
“Clean modern banner announcing local news in [CITY], warm tones, simple geometric shapes.”
“Minimalist community update graphic for [CITY], calm colors, subtle gradients.”

────────────────────────────────
FACEBOOK & REDDIT LINKS
────────────────────────────────
FACEBOOK GROUP LINK HARD-PROTECTION MODE (MANDATORY):
The group MUST be fully Public and viewable without login.
URL MUST follow: https://www.facebook.com/groups/[GROUPNAME]
ABSOLUTELY DO NOT return links with "?ref=", "/posts/", etc.

Format:
Facebook Groups (public):
• [Group Name] – [link] (Fully Verified Public Group – Login NOT required)

Reddit Communities:
• r/[SubName] – [link]

────────────────────────────────
OPERATION FLOW
────────────────────────────────
Every time the user provides a city:
1. Show Privacy Notice
2. Produce 3–5 community news items (Rules applied)
3. Give graphic idea + AI prompt for each
4. Provide extra generic city graphic prompts
5. Provide 3–5 verified public Facebook group links
6. Provide 2–4 public Reddit community links
7. End with closing phrase.

NEVER ask clarifying questions. NEVER delay. ALWAYS produce the full output immediately using the provided Search Data.

====================
LIVE SEARCH DATA (USE THIS TO FIND THE NEWS/LINKS):
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

# --- WEB SEARCH FUNCTION (SAFE VERSION) ---
def perform_web_search(query):
    """Uses DuckDuckGo to find real information."""
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        search = DuckDuckGoSearchRun()
        
        # Determine search strategy based on query context
        if "facebook" in query.lower() or "reddit" in query.lower():
             # Strategy for finding groups
             enhanced_query = f"{query}"
        else:
             # Strategy for finding news (Default)
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
    # Ava, Karina, and Troy need Creativity (0.7). Simon/Bob/Max need Precision (0.1).
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
            # KARINA: SEARCH FOR DISCUSSIONS
            status_placeholder = st.empty()
            status_placeholder.info(f"🔎 Karina is scanning for discussions in: {prompt}...")
            
            search_results = perform_web_search(f"site:reddit.com OR site:quora.com {prompt} real estate moving")
            status_placeholder.empty()
            
            system_instructions = base_prompt.replace("{user_raw_input}", "ANALYZE SEARCH RESULTS BELOW")
            full_user_message = f"USER QUERY: {prompt}\n\nSEARCH RESULTS:\n{search_results}"
            messages_payload = [SystemMessage(content=system_instructions), HumanMessage(content=full_user_message)]

        elif "Troy" in selected_agent:
            # TROY: SEARCH FOR NEWS & GROUPS
            if prompt.lower().strip() == "hello" or prompt.lower().strip() == "hi":
                # Si solo saluda, mostramos el mensaje de bienvenida sin buscar
                 messages_payload = [SystemMessage(content=base_prompt), HumanMessage(content=prompt)]
            else:
                status_placeholder = st.empty()
                status_placeholder.info(f"🗞️ Troy is gathering community news for: {prompt}...")
                
                # Troy hace 2 búsquedas: Una para noticias, otra para grupos
                news_results = perform_web_search(f"{prompt} community news new business opening events")
                social_results = perform_web_search(f"site:facebook.com/groups {prompt} public group")
                
                combined_results = f"NEWS RESULTS:\n{news_results}\n\nFACEBOOK/REDDIT RESULTS:\n{social_results}"
                status_placeholder.empty()
                
                system_instructions = base_prompt.replace("{user_raw_input}", "ANALYZE LIVE DATA BELOW")
                full_user_message = f"TARGET CITY: {prompt}\n\nLIVE WEB SEARCH DATA:\n{combined_results}"
                messages_payload = [SystemMessage(content=system_instructions), HumanMessage(content=full_user_message)]

        else:
            # Standard Logic
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
