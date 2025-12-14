# # --- LLM MODEL SETUP ---

# from typing import Any, Dict
# from huggingface_hub import InferenceClient
# import json
# from baseline import retrieve_from_kg
# from embeddings import semantic_search
# from preprocessing import get_neo4j_config

# CONFIG = get_neo4j_config()
# HF_TOKEN = CONFIG.get("HF_TOKEN")

# # REAL 3 DIFFERENT MODELS (FREE + GOOD + ACCURATE)
# LLM_MODELS_MAP = {
#     "Gemma-2B-IT": "google/gemma-2-2b-it",
#     "Mistral-7B-Instruct": "mistralai/Mistral-7B-Instruct-v0.2",
#     "Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct"
# }

# # Keep separate pipelines (so user can switch dynamically)
# LLM_PIPELINES = {}

# class HFInferenceWrapper:
#     def __init__(self, token: str, model_name: str):
#         self.client = InferenceClient(token=token, model=model_name)
#         self.model_name = model_name

#     def generate(self, prompt: str) -> str:
#         response = self.client.chat_completion(
#             model=self.model_name,
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=500,
#             temperature=0.2
#         )
#         return response.choices[0].message.content.strip()


# def get_llm_pipeline(model_label: str):
#     """Return cached instance OR create new one for selected model."""
#     global LLM_PIPELINES

#     if model_label in LLM_PIPELINES:
#         return LLM_PIPELINES[model_label]

#     model_name = LLM_MODELS_MAP[model_label]
#     print(f"🔍 Loading LLM model: {model_label} → {model_name}")

#     LLM_PIPELINES[model_label] = HFInferenceWrapper(
#         token=HF_TOKEN,
#         model_name=model_name
#     )

#     print(f"✅ Loaded: {model_label}")
#     return LLM_PIPELINES[model_label]


# # --- GENERATION STEP ---

# # --- GENERATION STEP (Prompt Fixed for Flexibility) ---

# def generate_response_from_results(question: str, pipeline_output: dict, selected_llm_label: str) -> str:
#     intent = pipeline_output.get("intent")
#     entities = pipeline_output.get("entities", {})

#     # Load the correct model depending on the user's selection
#     llm = get_llm_pipeline(selected_llm_label)

#     baseline_results = pipeline_output.get("baseline_results", [])
#     embedding_results = pipeline_output.get("embedding_results", [])

#     # Merge both retrievals into unified context
#     context = {
#         "baseline": baseline_results,
#         "semantic": embedding_results
#     }

#     persona = (
#         "You are an FPL expert assistant. "
#         "Use ONLY the provided knowledge graph context to answer. "
#         "Do not hallucinate. Be concise and correct."
#     )

#     task = (
#         f"Answer the user's question using ONLY the results in context. "
#         f"User Question: {question}"
#     )

#     # Logic to create dynamic, strict instructions
#     # has_baseline_data = baseline_results and "Error" not in baseline_results[0]
    
#     # if has_baseline_data:
#     #     # Flexible instruction to handle single facts (QA) or lists (Rankings)
#     #     instruction = (
#     #         "**STRICT INSTRUCTION: Use the results from the CONTEXT (KG Results) to directly answer the user's question.**\n"
#     #         "1. Focus ONLY on the results in the `context[\"baseline\"]` array.\n"
#     #         "2. **If the baseline array contains a list of multiple results (e.g., rankings, comparisons), format ALL results in the list as a neat, numbered list.**\n"
#     #         "3. **Maintain the EXACT ranking and order provided in the context (do not re-rank).**\n"
#     #         "4. **If the baseline array contains a single fact (e.g., one player's stats), state the fact conversationally without a list.**\n"
#     #         "5. Begin your answer with a brief, friendly conversational opening sentence.\n"
#     #         "6. DO NOT comment on the data quality or validity."
#     #     )
#     # else:
#     #     instruction = (
#     #         "**STRICT INSTRUCTION:**\n"
#     #         'If the `context["baseline"]` array is empty or contains errors, then and only then, state clearly: "The knowledge graph does not contain the answer."\n'
#     #         'DO NOT comment on the data quality or validity.'
#     #     )

#     has_baseline = bool(baseline_results) and "Error" not in baseline_results[0]
#     has_semantic = bool(embedding_results)

#     if has_baseline and has_semantic:
#         instruction = (
#         "Use BOTH context['baseline'] and context['semantic'].\n"
#         "Prefer baseline for exact numeric facts.\n"
#         "Use semantic to suggest relevant players when baseline is missing.\n"
#         "Do not hallucinate."
#         )
#     elif has_baseline:
#         instruction = (
#         "Use ONLY context['baseline'] to answer. Keep the given order."
#         )
#     elif has_semantic:
#         instruction = (
#         "Use ONLY context['semantic'] to answer. If the question needs exact stats "
#         "that are not present, say you can only provide similarity-based suggestions."
#         )
#     else:
#         instruction = (
#         'State: "The knowledge graph does not contain the answer."'
#         )


#     # Construct structured prompt
#     prompt = f"""
# ### PERSONA
# {persona}

# ### CONTEXT (KG Results)
# {json.dumps(context, indent=2)}

# ### TASK
# {task}

# {instruction}

# ### FINAL ANSWER:
# """

#     try:
#         # Call the generate method of the selected LLM pipeline
#         return llm.generate(prompt)
#     except Exception as e:
#         return f"LLM Error: {e}"


# # --- ORCHESTRATOR (unchanged except model selection is passed) ---

# def answer_query(query: str, intent: str, entities: Dict[str, Any], model_name: str,
#                  retrieval_method: str, embed_model_number: int) -> Dict[str, Any]:

#     baseline_results = []
#     baseline_cypher = None
#     embed_results = []

#     if retrieval_method in ["baseline", "hybrid"]:
#         baseline_results, baseline_cypher = retrieve_from_kg(intent, entities)

#     if retrieval_method in ["embeddings", "hybrid"]:
#         embed_results = semantic_search(query, model_number=embed_model_number, top_k=5)

#     pipeline_output = {
#         "question": query,
#         "intent": intent,
#         "entities": entities,
#         "baseline_results": baseline_results,
#         "embedding_results": embed_results,
#     }

#     final_answer = generate_response_from_results(query, pipeline_output, model_name)

#     return {
#         "answer": final_answer,
#         "baseline_results": baseline_results,
#         "baseline_cypher": baseline_cypher,
#         "embedding_results": embed_results,
#         "intent": intent,
#         "entities": entities
#     }










# llm_layer.py
# ===========================================================
#   FPL Graph-RAG — LLM Synthesis Layer (Option A: Baseline-first)
#   ✅ If baseline (Cypher) returns data -> final answer uses ONLY baseline
#   ✅ Semantic (embeddings) used ONLY if:
#        - embeddings-only mode, OR
#        - hybrid mode AND baseline returned empty/error
# ===========================================================

from typing import Any, Dict
from huggingface_hub import InferenceClient
import json

from baseline import retrieve_from_kg
from embeddings import semantic_search
from preprocessing import get_neo4j_config

# -----------------------------------------------------------
# Config
# -----------------------------------------------------------

CONFIG = get_neo4j_config()
HF_TOKEN = CONFIG.get("HF_TOKEN")

# REAL 3 DIFFERENT MODELS (FREE + GOOD + ACCURATE)
LLM_MODELS_MAP = {
    "Gemma-2B-IT": "google/gemma-2-2b-it",
    "Mistral-7B-Instruct": "mistralai/Mistral-7B-Instruct-v0.2",
    "Llama-3.1-8B-Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct"
}

# Cached pipelines
LLM_PIPELINES: Dict[str, "HFInferenceWrapper"] = {}


class HFInferenceWrapper:
    def __init__(self, token: str, model_name: str):
        self.client = InferenceClient(token=token, model=model_name)
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        response = self.client.chat_completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()


def get_llm_pipeline(model_label: str) -> HFInferenceWrapper:
    """Return cached instance OR create new one for selected model."""
    global LLM_PIPELINES

    if model_label in LLM_PIPELINES:
        return LLM_PIPELINES[model_label]

    if model_label not in LLM_MODELS_MAP:
        raise ValueError(f"Unknown model label: {model_label}")

    model_name = LLM_MODELS_MAP[model_label]
    print(f"🔍 Loading LLM model: {model_label} → {model_name}")

    if not HF_TOKEN:
        raise ValueError("HF_TOKEN is missing in config.txt")

    LLM_PIPELINES[model_label] = HFInferenceWrapper(
        token=HF_TOKEN,
        model_name=model_name
    )

    print(f"✅ Loaded: {model_label}")
    return LLM_PIPELINES[model_label]


# -----------------------------------------------------------
# Generation Step (Baseline-first: prevents embedding pollution)
# -----------------------------------------------------------

def generate_response_from_results(question: str, pipeline_output: dict, selected_llm_label: str) -> str:
    """
    Creates a strict prompt so the LLM:
    - Uses baseline results ONLY if present (correct for rankings / exact stats)
    - Uses embeddings ONLY when baseline fails or embeddings-only mode is selected
    """
    llm = get_llm_pipeline(selected_llm_label)

    baseline_results = pipeline_output.get("baseline_results", []) or []
    embedding_results = pipeline_output.get("embedding_results", []) or []
    retrieval_method = pipeline_output.get("retrieval_method", "hybrid")

    # Baseline validity check
    has_baseline_data = bool(baseline_results) and not (
        isinstance(baseline_results[0], dict) and "Error" in baseline_results[0]
    )

    has_semantic_data = bool(embedding_results)

    # ✅ Option A rule:
    # Use semantic only if embeddings-only OR hybrid and baseline failed
    use_semantic = (retrieval_method == "embeddings") or (retrieval_method == "hybrid" and not has_baseline_data)

    # Build context: keep semantic visible only if we intend to use it in the answer
    context = {
        "baseline": baseline_results,
        "semantic": embedding_results if use_semantic else []
    }

    persona = (
        "You are an FPL expert assistant. "
        "Use ONLY the provided context to answer. "
        "Do not hallucinate. Be concise and correct."
    )

    task = (
        f"Answer the user's question using ONLY the results in context.\n"
        f"User Question: {question}"
    )

    # -----------------------------
    # Strict instruction policy
    # -----------------------------
    # Unified Hybrid Instruction (80% Baseline / 20% Semantic)
    if has_baseline_data or has_semantic_data:
        instruction = (
            "**STRICT INSTRUCTION (HYBRID RAG AGGREGATION):**\n"
            "1) **PRIORITIZE BASELINE (80% Weight):** Use `context['baseline']` for all hard facts, exact stats, and official rankings. "
            "The Baseline represents the ground truth of the Knowledge Graph.\n"
            "2) **AUGMENT WITH SEMANTIC (20% Weight):** Use `context['semantic']` ONLY to add context, 'similar-style' suggestions, "
            "or to fulfill conceptual requests (like 'Who is like this player?') that Baseline can't capture.\n"
            "3) **CONFLICT RESOLUTION:** If Baseline and Semantic provide different numbers for the same player, ALWAYS use the Baseline number.\n"
            "4) **LIST FORMATTING:** If answering a list query (e.g., Top 10), provide the official Baseline list first. "
            "Then, add a 'Semantic Suggestions' section at the end for players who fit the vibe but weren't in the top stat list.\n"
            "5) **TRUTHFULNESS:** If Baseline is empty but Semantic has data, provide the suggestions but clearly state: "
            "'Exact database records were not found, but based on playing style, here are the most relevant players:'\n"
            "6) **NO GHOSTING:** Do not mention internal context names like 'context[baseline]'. Answer as an FPL expert."
        )

    else:
        # No usable data
        instruction = (
            "**STRICT INSTRUCTION:**\n"
            'State exactly: "The knowledge graph does not contain the answer."\n'
            "Do not add anything else."
        )

    # Construct prompt
    prompt = f"""
### PERSONA
{persona}

### CONTEXT (KG Results)
{json.dumps(context, indent=2)}

### TASK
{task}

{instruction}

### FINAL ANSWER:
"""

    try:
        return llm.generate(prompt)
    except Exception as e:
        return f"LLM Error: {e}"


# -----------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------

def answer_query(
    query: str,
    intent: str,
    entities: Dict[str, Any],
    model_name: str,
    retrieval_method: str,
    embed_model_number: int
) -> Dict[str, Any]:

    baseline_results = []
    baseline_cypher = None
    embed_results = []

    # 1) Baseline retrieval
    if retrieval_method in ["baseline", "hybrid"]:
        baseline_results, baseline_cypher = retrieve_from_kg(intent, entities)

    # 2) Embedding retrieval
    if retrieval_method in ["embeddings", "hybrid"]:
        embed_results = semantic_search(query, model_number=embed_model_number, top_k=5)

    # ✅ IMPORTANT: pass retrieval_method into the pipeline so generation can enforce Option A
    pipeline_output = {
        "question": query,
        "intent": intent,
        "entities": entities,
        "baseline_results": baseline_results,
        "embedding_results": embed_results,
        "retrieval_method": retrieval_method
    }

    final_answer = generate_response_from_results(query, pipeline_output, model_name)

    return {
        "answer": final_answer,
        "baseline_results": baseline_results,
        "baseline_cypher": baseline_cypher,
        "embedding_results": embed_results,
        "intent": intent,
        "entities": entities
    }
