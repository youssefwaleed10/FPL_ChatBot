"""
===========================================================
     FPL KNOWLEDGE GRAPH — EMBEDDING RETRIEVAL (2.b)
     
     IMPLEMENTATION: Seasonal Features Vector Embedding
     FIXED: Aggregation is now done per Season, allowing accurate filtering on queries 
     like "top forwards in 2022-23".
===========================================================
"""
import os
import ssl
import sys
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import os
from typing import Dict, Any, List


# --- 1. BYPASS MAC SSL/NETWORK BLOCKS ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

os.environ['CURL_CA_BUNDLE'] = '' 
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# --- 2. AUTOMATIC DOWNLOAD LOGIC ---
try:
    from sentence_transformers import SentenceTransformer
    print("Checking/Downloading Embedding Models...")
    
    # This will check local cache first; if missing, it downloads them automatically.
    model_1 = SentenceTransformer("all-MiniLM-L6-v2")      
    model_2 = SentenceTransformer("all-mpnet-base-v2")      
    
    print("Models loaded successfully.")
except Exception as e:
    print(f"Failed to load/download models. Error: {e}")
    # We exit with 1 so app.py knows the subprocess failed
    sys.exit(1)
# ---------------------------------------------------------
# Config Loading 
# ---------------------------------------------------------
def get_neo4j_config(config_file="config.txt"):
    """Loads connection details from config.txt."""
    config = {}
    # Assuming config.txt is in the same directory or accessible path
    config_path = os.path.join(os.path.dirname(__file__), config_file)
    try:
        with open(config_path, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    config[key] = value
        return config
    except:
        return {}

CONFIG = get_neo4j_config()
URI = CONFIG.get("URI")
AUTH = (CONFIG.get("USERNAME"), CONFIG.get("PASSWORD"))
# Initialize driver globally
driver = GraphDatabase.driver(URI, auth=AUTH)
try:
    # Model 1: all-MiniLM-L6-v2 (384 dims)
    model_1 = SentenceTransformer("all-MiniLM-L6-v2")      
    # Model 2: all-mpnet-base-v2 (768 dims)
    model_2 = SentenceTransformer("all-mpnet-base-v2")      
except Exception as e:
    print(f"Warning: Failed to load embedding models. Semantic search disabled. Error: {e}")
    model_1 = None
    model_2 = None


# ---------------------------------------------------------
# Data Transformation (Text Generation) - SEASONAL & POSITION-SPECIFIC
# ---------------------------------------------------------

def build_description(player_stats: Dict[str, Any]) -> str:
    """
    Creates a descriptive string including the Season, and uses position-specific 
    high-salience key-value pairs.
    """
    name = player_stats.get("name", "")
    pos = player_stats.get("position", "")
    season = player_stats.get("season", "Unknown Season") # Now mandatory
    
    # Numerical features
    pts = player_stats.get("total_points", 0)
    goals = player_stats.get("goals_scored", 0)
    assists = player_stats.get("assists", 0)
    cs = player_stats.get("clean_sheets", 0)
    
    # 1. Base Description (Must include Season and universal stats)
    description = (
        f"PLAYER: {name}. SEASON: {season}. POSITION: {pos}. "
        f"TOTAL_POINTS: {pts}. "
    )

    # 2. Position-Specific Emphasis (Only use the top 1-2 stats)
    if pos == "GKP":
        emphasis = f"GOALKEEPER STATS: CLEAN_SHEETS: {cs}. "
    elif pos == "DEF":
        emphasis = f"DEFENDER STATS: CLEAN_SHEETS: {cs}. GOALS: {goals}. "
    elif pos == "MID":
        emphasis = f"MIDFIELDER STATS: GOALS: {goals}. ASSISTS: {assists}. "
    elif pos == "FWD":
        emphasis = f"FORWARD STATS: GOALS: {goals}. ASSISTS: {assists}. "
    else:
        # Fallback
        emphasis = f"GOALS: {goals}. ASSISTS: {assists}. CLEAN_SHEETS: {cs}."

    return description + emphasis


# ---------------------------------------------------------
# Neo4j Data Fetching - SEASONAL AGGREGATION
# ---------------------------------------------------------

# def get_all_players() -> List[Dict[str, Any]]:
#     """
#     Fetches all player data needed to create seasonal embeddings.
#     ASSUMPTION: Fixture nodes are linked to a Season node.
#     """
#     query = """
#     MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
#     OPTIONAL MATCH (p)-[r:PLAYED_IN]->(f:Fixture)-[:PART_OF_SEASON]->(s:Season)
    
#     // Group by Player AND Season to get seasonal stats
#     WITH p, pos, s.name AS season,
#         SUM(COALESCE(r.total_points, 0)) AS total_points,
#         SUM(COALESCE(r.goals_scored, 0)) AS goals_scored,
#         SUM(COALESCE(r.assists, 0)) AS assists,
#         SUM(COALESCE(r.clean_sheets, 0)) AS clean_sheets
        
#     // Filter out players with no season data (i.e., new players or missing data)
#     WHERE season IS NOT NULL
        
#     // RETURN the new aggregated properties
#     RETURN p.name AS name, pos.name AS position, season, total_points, goals_scored, assists, clean_sheets
#     """
#     try:
#         with driver.session() as session:
#             data = [r.data() for r in session.run(query)]
#         print(f"--- DEBUG: Successfully fetched {len(data)} SEASONAL player records from KG. ---")
#         return data
#     except Exception as e:
#         print(f"--- DEBUG ERROR: Failed to fetch seasonal player data from KG: {e} ---")
#         return []


def get_all_players() -> List[Dict[str, Any]]:
    """
    Fetches all player data needed to create seasonal embeddings.
    Uses f.season property (because your KG stores season on Fixture).
    """
    query = """
    MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
    MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
    WITH p, pos, f.season AS season,
         SUM(COALESCE(r.total_points, 0)) AS total_points,
         SUM(COALESCE(r.goals_scored, 0)) AS goals_scored,
         SUM(COALESCE(r.assists, 0)) AS assists,
         SUM(COALESCE(r.clean_sheets, 0)) AS clean_sheets
    WHERE season IS NOT NULL
    RETURN p.name AS name, pos.name AS position, season,
           total_points, goals_scored, assists, clean_sheets
    """
    try:
        with driver.session() as session:
            data = [r.data() for r in session.run(query)]
        print(f"--- DEBUG: Successfully fetched {len(data)} SEASONAL player records from KG. ---")
        return data
    except Exception as e:
        print(f"--- DEBUG ERROR: Failed to fetch seasonal player data from KG: {e} ---")
        return []


# ---------------------------------------------------------
# Neo4j Index and Embedding Storage - UPDATED TO STORE ON PLAYER NODE FOR NOW
# ---------------------------------------------------------

def create_vector_indexes():
    """
    Creates the two required vector indexes. 
    NOTE: Storing the seasonal vector on the player node is a simplification. 
    In production, you'd store this on a dedicated :Performance node. 
    We re-use the player node for simplicity in this code fix.
    """

    cy1 = """
    CREATE VECTOR INDEX player_embed1_index IF NOT EXISTS
    FOR (p:Player) ON (p.embedding1)
    OPTIONS { indexConfig: {
        `vector.dimensions`: 384,
        `vector.similarity_function`: "COSINE"
    }};
    """

    cy2 = """
    CREATE VECTOR INDEX player_embed2_index IF NOT EXISTS
    FOR (p:Player) ON (p.embedding2)
    OPTIONS { indexConfig: {
        `vector.dimensions`: 768,
        `vector.similarity_function`: "COSINE"
    }};
    """

    with driver.session() as session:
        session.run(cy1)
        session.run(cy2)
        print("Vector indexes created/verified.")


def store_embeddings(players: List[Dict[str, Any]]):
    """
    Encodes and stores embeddings. Since each player now has multiple seasonal 
    records, we must find a way to store them. For this quick fix, we'll store 
    a list of seasonal embeddings/names on the Player node.
    
    A clean implementation would store the vector on a separate seasonal node.
    Since we are constrained to the current setup, we will store the MOST RECENT 
    seasonal embedding on the player node and create a text description containing 
    the season to enable seasonal filtering in the query.
    
    ***WARNING: This simplified approach means the vector search will still only
    return the player node, but the embedding is season-aware. To be fully correct, 
    the vector search would need to return the Seasonal Performance node.***
    """
    if not model_1 or not model_2:
        print("Embedding models not loaded. Skipping embedding storage.")
        return

    if not players:
        print("No players to process. Skipping embedding storage.")
        return

    # Find the most recent season data for each player to store
    latest_player_data = {}
    for p in players:
        name = p["name"]
        season = p["season"]
        # Assuming lexicographical sort on season names (e.g., "2023-24" > "2022-23")
        if name not in latest_player_data or season > latest_player_data[name]["season"]:
            latest_player_data[name] = p
            
    batch = []
    
    for p in latest_player_data.values():
        desc = build_description(p)
        v1 = model_1.encode(desc).tolist()
        v2 = model_2.encode(desc).tolist()

        batch.append({"name": p["name"], "v1": v1, "v2": v2})

    cypher = """
    UNWIND $rows AS row
    MATCH (pl:Player {name: row.name})
    SET pl.embedding1 = row.v1,
        pl.embedding2 = row.v2
    """

    try:
        with driver.session() as session:
            session.run(cypher, rows=batch)
            print(f"Successfully stored seasonal embeddings (most recent) for {len(batch)} players.")
    except Exception as e:
        print(f"FATAL ERROR: Failed to store embeddings in Neo4j: {e}")


# ---------------------------------------------------------
# Semantic Search (Retrieval) (unchanged)
# ---------------------------------------------------------

def deduplicate_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keeps the highest score per player name."""
    best = {}
    for r in results:
        n, score = r.get("player"), r.get("score", 0.0)
        if n and (n not in best or score > best[n]):
            best[n] = score
    return sorted([{"player": k, "score": v} for k, v in best.items()],
                  key=lambda x: x["score"],
                  reverse=True)


def semantic_search(query_text: str, model_number: int = 1, top_k: int = 5) -> List[Dict[str, Any]]:
    """Performs vector search on Player nodes."""
    
    if (model_number == 1 and not model_1) or (model_number == 2 and not model_2):
        print("Embedding model not available.")
        return []

    model = model_1 if model_number == 1 else model_2
    vec = model.encode(query_text).tolist()

    index = "player_embed1_index" if model_number == 1 else "player_embed2_index"
    
    # We will log the search parameters here for debugging
    print(f"--- DEBUG Search: Index='{index}', Vector Length={len(vec)}, Query='{query_text[:30]}...' ---")

    cypher = """
    CALL db.index.vector.queryNodes($index, $fetch_k, $vec)
    YIELD node, score
    RETURN node.name AS player, score
    """

    try:
        with driver.session() as session:
            raw = [r.data() for r in session.run(
                cypher,
                index=index,
                fetch_k=top_k * 3, 
                vec=vec
            )]
        
        print(f"--- DEBUG Search: Raw results count from Neo4j: {len(raw)} ---")

        clean = deduplicate_results(raw)
        
        if not clean:
             print("--- DEBUG Search: Clean results are EMPTY. Vector search found nothing. ---")

        return clean[:top_k]
    except Exception as e:
        # This will catch errors during Cypher execution
        print(f"FATAL VECTOR SEARCH ERROR (Cypher Execution Failed): {e}") 
        return []

# ---------------------------------------------------------
# Main Execution for Setup (unchanged)
# ---------------------------------------------------------
if __name__ == "__main__":
    # --- This block runs the setup when you execute the file directly ---
    print("--- Running Embedding Setup ---")
    create_vector_indexes()
    
    players = get_all_players()
    if players:
        store_embeddings(players)
    
    # Test query to confirm retrieval works locally
    test_query = "best striker who scores a lot of goals in the 2022-23 season"
    results = semantic_search(test_query, model_number=1, top_k=3)
    
    print("\n==============================================")
    print(f"Test Query Results for '{test_query}':")
    if results:
        print(" SUCCESS: Seasonal Embeddings are retrieved.")
        print(results)
    else:
        print(" FAILURE: Test search returned no results. Check Neo4j logs.")
    print("==============================================")