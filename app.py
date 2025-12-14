import streamlit as st
from llm_layer import answer_query
from preprocessing import extract_entities, classify_intent
import networkx as nx
import matplotlib.pyplot as plt

# -------------------------------------------------------
# Page Setups
# -------------------------------------------------------
st.set_page_config(
    page_title="FPL Graph-RAG Assistant",
    layout="wide"
)

st.title("⚽ FPL Graph-RAG Assistant")
st.write("An end-to-end Graph-RAG system for Fantasy Premier League analysis. It combines symbolic (Cypher) and statistical (Embeddings) retrieval with an LLM for factual, insightful answers.")

# Initialize session state for consistent query handling
if "user_query" not in st.session_state:
    st.session_state["user_query"] = ""

# -------------------------------------------------------
# SIDEBAR (LEFT PANEL)
# -------------------------------------------------------
st.sidebar.title("⚙️ System Configuration")

# Retrieval method sidebar
retrieval_method = st.sidebar.selectbox(
    "1. Retrieval Method",
    ["hybrid", "baseline", "embeddings"],
    help="Hybrid: Use Cypher + Embeddings. Baseline: Cypher only. Embeddings: Vector search only."
)

# LLM model selection (Updated to Google/Gemma models for comparison)
# model_name = st.sidebar.selectbox(
#     "2. Choose LLM Model (Comparison required)",
#     ["Gemma-2B-IT", "Gemma-7B-IT", "Gemini-API"], 
#     help="Select a model for the final answer synthesis. The system uses a Gemma API client, with the label passed to the RAG prompt for comparison."
# )
model_name = st.sidebar.selectbox(
    "2. Choose LLM Model (Comparison required)",
    ["Gemma-2B-IT", "Mistral-7B-Instruct", "Llama-3.1-8B-Instruct"],
    help="Select a model for the final answer synthesis. The system uses a Gemma API client, with the label passed to the RAG prompt for comparison."
)

# Embedding Model Selection
embed_model_selection = st.sidebar.selectbox(
    "3. Embedding Model (if using Embeddings)",
    [1, 2], 
    format_func=lambda x: f"Model {x}",
    help="1: all-MiniLM-L6-v2, 2: all-mpnet-base-v2."
)

st.sidebar.write("---")

# -------------------------------------------------------
# Suggested Questions 
# -------------------------------------------------------
# suggestions = [
#     "How many points did Mohamed Salah get in GW5?",
#     "Top forwards in 2022-23",
#     "Recommend a cheap midfielder for gameweek 7",
#     "Compare Erling Haaland and Heung-Min Son",
#     "Show me Man City's upcoming fixtures"
# ]
suggestions = [
    "How many points, goals, assists, and minutes did Mohamed Salah get in Gameweek 5 of the 2022–23 season?",
    "Who are the top 10 players by total points in the 2022–23 season?",
    "Who are the top 10 midfielders in the 2022–23 season based on total points?",
    "Compare Mohamed Salah and Bukayo Saka in terms of total points, goals, and assists during the 2022–23 season?",
    # --- REVISED QUESTION 1: Added explicit limit for robustness ---
    "Who are the top 10 players most similar to Mohamed Salah in the 2022-23 season?",
    # --- REVISED QUESTION 2: Explicitly requests "first 5" ---
    "Show me Mohamed Salah's last 5 fixtures and the points he scored in each during the 2022-23 season.",
    # --- REVISED QUESTION 3: Added explicit limit for robustness ---
    "List the top 10 midfielders by points per 90 minutes in the 2022-23 season.",
    # --- REVISED QUESTION 4: Rephrased for clear team analysis intent ---
    # "What were the total goals conceded by Liverpool and the number of matches they played in the 2022-23 season?",
    "Who are the top 10 players with the most bonus points in the 2022–23 season?",
    "In which gameweeks did Mohamed Salah score more points than his season average in the 2022–23 season?",
    "What are Arsenal’s upcoming fixtures starting from Gameweek 25 in the 2022–23 season?",
    "Who are the top 10 players with the highest goal contributions (goals plus assists) in the 2022–23 season?"
]


st.subheader("💡 Suggested Questions")
cols = st.columns(2)

for i, q in enumerate(suggestions):
    with cols[i % 2]:
        # Use a consistent button to update state and rerun
        if st.button(q, use_container_width=True, key=f"btn_{i}"):
            st.session_state["user_query"] = q
            st.rerun() 

# -------------------------------------------------------
# User Input Area
# -------------------------------------------------------
st.subheader("🔍 Ask your FPL question")
query = st.text_area(
    "Enter your question:",
    value=st.session_state.get("user_query", ""),
    height=100
)

run_button = st.button("Run", type="primary")

# -------------------------------------------------------
# PROCESS PIPELINE
# -------------------------------------------------------
if run_button and query.strip():

    # Clear previous query state after execution
    st.session_state["user_query"] = query
    
    with st.spinner('Running...'):
        
        # 1. Preprocessing
        st.divider()
        st.subheader("🧠 Understanding Your Query (Step 1: Preprocessing)")

        intent = classify_intent(query)
        entities = extract_entities(query)

        st.markdown(f"**Detected Intent:** `{intent}`")
        st.markdown(f"**Extracted Entities:** `{entities}`")
        
        # 2. Run LLM Layer (which orchestrates retrieval and generation)
        try:
            result = answer_query(
                query=query, 
                intent=intent, 
                entities=entities, 
                model_name=model_name, # <-- The selected model name is passed here
                retrieval_method=retrieval_method,
                embed_model_number=embed_model_selection
            )
            
            baseline_results = result.get("baseline_results", [])
            baseline_cypher = result.get("baseline_cypher")
            embed_results = result.get("embedding_results", [])
            final_answer = result.get("answer", "Error: LLM failed to generate an answer.")
        
        except Exception as e:
            st.error(f"An error occurred during the RAG process: {e}")
            final_answer = f"System Error: {e}"
            baseline_results = []
            baseline_cypher = None
            embed_results = []


    # ------------------ CONTEXT SECTION (Step 2: Retrieval) ------------------
    st.divider()
    st.subheader("📦 KG Retrieved Context")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("### 🔷 Baseline (Cypher) Results")
        if baseline_results and "Error" not in baseline_results[0]:
            st.json(baseline_results)
        elif retrieval_method in ["baseline", "hybrid"]:
            st.warning("No data retrieved via Cypher or an error occurred.")

    with colB:
        st.markdown(f"### 🔶 Embedding (Model {embed_model_selection}) Results")
        if embed_results:
            st.json(embed_results)
        elif retrieval_method in ["embeddings", "hybrid"]:
            st.warning("No data retrieved via Embeddings.")

    # ------------------ CYPHER QUERY ------------------
    st.divider()
    st.subheader("📝 Executed Cypher Query")
    if baseline_cypher:
        st.code(baseline_cypher, language="cypher")
    else:
        st.write("No Cypher query executed (Embedding-only mode or no template match).")

    # ------------------ GRAPH VISUALIZATION ------------------
    st.divider()
    st.subheader("📊 Graph Visualization Snippet")

    try:
        G = nx.Graph()
        if baseline_results and "Error" not in baseline_results[0]:
            for record in baseline_results:
                player_name = record.get("player", "N/A")
                G.add_node(player_name, label="Player", color="skyblue")
                
                # Add edges for relevant properties
                for k, v in record.items():
                    if k not in ["player", "season"] and v is not None and isinstance(v, (str, int, float)):
                        stat_label = f"{k}: {v}"
                        G.add_node(stat_label, label=k, color="lightgreen", size=100)
                        G.add_edge(player_name, stat_label)
                        
            if G.number_of_nodes() > 1:
                plt.figure(figsize=(5, 3))
                node_colors = [G.nodes[n].get('color', 'grey') for n in G.nodes()]
                pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42) 
                
                nx.draw(G, pos, with_labels=True, node_size=400, node_color=node_colors, font_size=5, font_weight="bold", edge_color='gray')
                st.pyplot(plt)
                plt.close() # Close plot figure to prevent memory issues
            else:
                 st.write("Graph visualization requires a successful Cypher retrieval with structured data.")
        else:
             st.write("No baseline results to visualize.")

    except Exception as e:
        st.write(f"Visualization skipped due to error: {e}")

    # ------------------ FINAL ANSWER (Step 3: Generation) ------------------
    st.divider()
    st.subheader(f"🎯 FINAL ANSWER (Model: {model_name})")
    
    st.markdown(
        f"""
        <div style="padding: 20px; background-color: #000000; border-radius: 10px; border-left: 6px solid #4CAF50; border: 1px solid #4CAF50;">
            {final_answer}
        </div>
        """,
        unsafe_allow_html=True
    )
    st.success("Analysis complete!")