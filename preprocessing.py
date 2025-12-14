import re
import unicodedata
import os
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# Neo4j Config Loader
# ---------------------------------------------------------
def get_neo4j_config(config_file="config.txt"):
    config = {}
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

# ---------------------------------------------------------
# INTENT CLASSIFICATION
# ---------------------------------------------------------

INTENT_KEYWORDS = {
    # 1. QA – Player stats in a specific GW (Focus on 'stats' + 'GW')
    "qa_stat_query": [
        r"points.*gameweek",
        r"stats.*gameweek",
        r"performance in gameweek",
        r"how many goals did .* get .* gameweek", # Added to capture specifics
        r"stats of .* in gw"
    ],

        # 7. Differential / advanced performance (points per 90)
   "points_per_90": [
    r"points per 90", r"per 90 minutes", 
    r"best per 90", r"highest per 90", 
    r"midfielders by points per 90", # <--- Add this
    r"players by points per 90"      # <--- And this, for robustness
    ],

    # 2. Top players overall (season) - This should only trigger if NO position is mentioned.
    "top_players": [
        r"top \d+ players by total points",
        r"overall rankings"
    ],

    # 3. Top players by position (Highly specific)
    "top_players_by_position": [
        r"top .* midfielders",
        r"top .* defenders",
        r"top .* forwards",
        r"best .* by position"
    ],

    # 4. Compare players
    "compare_players": [
        r"\bcompare\b", r"\bvs\b", r"difference between", r"compare .* and .*"
    ],

    # 5. Similar players (Crucial to capture 'similar')
    "similar_players": [
        r"similar players", r"players similar to", r"who is similar to", r"alternative to", r"most similar to"
    ],

    # 6. Fixtures of a player (Focus on 'fixtures' AND a 'player')
    "fixtures_of_player": [
        r"fixtures of .* player",
        r"matches of .* player",
        r"last (?:five|5|ten|10) fixtures", # Specific structure
        r"games played by"
    ],

#     # 7. Differential / advanced performance (points per 90)
#    "points_per_90": [
#     r"points per 90", r"per 90 minutes", 
#     r"best per 90", r"highest per 90", 
#     r"midfielders by points per 90", # <--- Add this
#     r"players by points per 90"      # <--- And this, for robustness
# ],

    # 8. Team analysis – defensive stats (Focus on 'concede' + 'goals')
    "team_analysis": [
        r"goals conceded",
        r"how many goals did .* concede",
        r"defensive performance"
    ],

    #8. BPS
    "top_bonus_points": [
    r"bonus points",
    r"most bonus",
    r"highest bonus"
    ],


    # 9. Above-average performances (Crucial to capture 'average')
    "above_average_performance": [
        r"above average", r"better than average", r"more points than his season average" # Added exact phrase
    ],

    # 10. Upcoming fixtures for a team
    "upcoming_team_fixtures": [
        r"upcoming fixtures", r"next fixtures", r"future matches", r"next games",
        r"starting from gameweek", # Crucial addition
        r"fixtures of .* team"
    ],

    # 11. Top goal contributors (Crucial to capture 'contributions' or 'goals plus assists')
    "top_goal_contributors": [
        r"goal contributions",
        r"goals plus assists",
        r"highest goal contributions",
        r"most goals and assists"
    ],
    
}

def classify_intent(text):
    txt = text.lower()
    for intent, patterns in INTENT_KEYWORDS.items():
        for p in patterns:
            if re.search(p, txt):
                return intent
    return "unknown"

# ---------------------------------------------------------
# LOAD ENTITY LISTS FROM NEO4J (with fallback)
# ---------------------------------------------------------

PLAYERS_LIST = []
TEAMS_LIST = []

def load_entities():
    global PLAYERS_LIST, TEAMS_LIST

    config = get_neo4j_config()
    URI = config.get("URI")
    AUTH = (config.get("USERNAME"), config.get("PASSWORD"))


    if not all([URI, AUTH[0], AUTH[1]]):
        PLAYERS_LIST = [
            "Erling Haaland", "Mohamed Salah", "Bukayo Saka",
            "Heung-Min Son", "Kevin De Bruyne", "Phil Foden"
        ]
        TEAMS_LIST = ["Arsenal", "Liverpool", "Manchester City", "Chelsea"]
        return

    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(URI, auth=AUTH)

        q1 = "MATCH (p:Player) RETURN DISTINCT p.name AS name"
        q2 = "MATCH (t:Team) RETURN DISTINCT t.name AS name"

        with driver.session() as session:
            PLAYERS_LIST = [r["name"] for r in session.run(q1) if r["name"]]
            TEAMS_LIST = [r["name"] for r in session.run(q2) if r["name"]]

        driver.close()
    except:
        PLAYERS_LIST = [
            "Erling Haaland", "Mohamed Salah",
            "Bukayo Saka", "Heung-Min Son"
        ]
        TEAMS_LIST = ["Arsenal", "Liverpool", "Manchester City"]

load_entities()

# ---------------------------------------------------------
# ALIAS MAPPING (NEW)
# ---------------------------------------------------------

PLAYER_ALIAS_MAP = {
    "salah": "Mohamed Salah",
    "haaland": "Erling Haaland",
    "kdb": "Kevin De Bruyne",
    "foden": "Phil Foden",
    "saka": "Bukayo Saka"
}

TEAM_ALIAS_MAP = {
    "man city": "Manchester City",
    "mancity": "Manchester City",
    "city": "Manchester City",
    "liverpool": "Liverpool",
    "arsenal": "Arsenal",
    "chelsea": "Chelsea",
    "man u": "Manchester Utd",
    "man utd": "Manchester Utd",
}

# ---------------------------------------------------------
# TEXT NORMALIZATION
# ---------------------------------------------------------

def normalize_text(s):
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---------------------------------------------------------
# PLAYER EXTRACTION
# ---------------------------------------------------------

def extract_players(text):
    txt = normalize_text(text)
    found = []

    # alias detection
    for alias, real in PLAYER_ALIAS_MAP.items():
        if alias in txt:
            found.append(real)

    # direct detection
    for p in PLAYERS_LIST:
        if normalize_text(p) in txt:
            found.append(p)

    return list(set(found))

# ---------------------------------------------------------
# TEAM EXTRACTION
# ---------------------------------------------------------

def extract_teams(text):
    txt = normalize_text(text)
    found = []

    for alias, real in TEAM_ALIAS_MAP.items():
        if alias in txt:
            found.append(real)

    for t in TEAMS_LIST:
        if t.lower() in txt:
            found.append(t)

    return list(set(found))

# ---------------------------------------------------------
# POSITION EXTRACTION
# ---------------------------------------------------------

POSITION_KEYWORDS = {
    "FWD": ["forward", "striker"],
    "MID": ["midfielder", "mid"],
    "DEF": ["defender", "def"],
    "GKP": ["goalkeeper", "keeper", "gk"]
}

def extract_positions(text):
    lower = text.lower()
    for pos, words in POSITION_KEYWORDS.items():
        if any(w in lower for w in words):
            return pos
    return None

# ---------------------------------------------------------
# SEASON EXTRACTION (WITH GENERALIZATION)
# ---------------------------------------------------------

def find_year(text):
    txt = text.lower()

    if "this season" in txt:
        return "2022-23"
    if "last season" in txt or "previous season" in txt:
        return "2021-22"

    # explicit season "2022-23"
    m = re.search(r"\b(20\d{2}-\d{2})\b", txt)
    if m:
        return m.group(1)

    # numeric year only
    m2 = re.search(r"\b(20\d{2})\b", txt)
    if m2:
        year = int(m2.group(1))
        end = (year % 100) + 1
        return f"{year}-{end:02d}"

    return "2022-23"  # default

# ---------------------------------------------------------
# GAMEWEEK EXTRACTION (GENERALIZED)
# ---------------------------------------------------------

def find_gameweek(text):
    txt = text.lower()

    if "this gw" in txt or "current gw" in txt:
        return 38
    if "previous gw" in txt:
        return 38

    m = re.search(r"(?:gw|gameweek|game week)\s*(\d+)", txt)
    return int(m.group(1)) if m else None

# ---------------------------------------------------------
# ATTRIBUTE EXTRACTION (CRITICAL FOR 11 SCENARIOS)
# ---------------------------------------------------------

def find_attributes(text):
    txt = text.lower()
    attributes = []

    if "cheap" in txt or "budget" in txt:
        attributes.append("cheap")

    if "fixtures" in txt or "matches" in txt:
        attributes.append("fixtures")

    if "form" in txt or "recent form" in txt:
        attributes.append("form")

    if "improved" in txt or "most improved" in txt:
        attributes.append("improved")

    if "bps" in txt or "bonus points" in txt:
        attributes.append("bps")

    if "differential" in txt or "low ownership" in txt:
        attributes.append("differential")

    return attributes

# ---------------------------------------------------------
# LIMIT EXTRACTION
# ---------------------------------------------------------

# In preproccessing.py, update the 'extract_limit' function:

# In preproccessing.py, this is the definitive extract_limit function:
def extract_limit(text):
    txt = text.lower()
    
    # Map spelled-out numbers
    num_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    
    # 1. Look for 'top N', 'best N' (e.g., top 10 players)
    m = re.search(r"(top|best|show|find|highest)\s+(\d+|" + "|".join(num_map.keys()) + r")", txt)
    if m:
        # Check if the limit is a number or a spelled-out word
        limit_str = m.group(2)
        return int(limit_str) if limit_str.isdigit() else num_map.get(limit_str, 10)

    # 2. Look for 'first N' or 'last N' fixtures/results (e.g., first 5 fixtures)
    m2 = re.search(r"(?:first|last)\s+(\d+|" + "|".join(num_map.keys()) + r")", txt)
    if m2:
        # Check if the limit is a number or a spelled-out word
        limit_str = m2.group(1)
        return int(limit_str) if limit_str.isdigit() else num_map.get(limit_str, 5) # Default to 5 for fixture/form lookups
        
    return None
# ---------------------------------------------------------
# MAIN ENTITY EXTRACTION
# ---------------------------------------------------------

def extract_entities(text):
    attributes = find_attributes(text)

    return {
        "players": extract_players(text),
        "teams": extract_teams(text),
        "season": find_year(text),
        "gameweek": find_gameweek(text),
        "position": extract_positions(text),
        "attributes": attributes,
        "limit": extract_limit(text),
        # "misc_keywords": text
    }

# ---------------------------------------------------------
# EMBEDDING LAYER (UNCHANGED)
# ---------------------------------------------------------

MODEL_1_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_2_NAME = "sentence-transformers/sentence-t5-base"

try:
    model1 = SentenceTransformer(MODEL_1_NAME)
    model2 = SentenceTransformer(MODEL_2_NAME)
except:
    model1 = None
    model2 = None

def embed_query_text(text, model_id=1):
    model = model1 if model_id == 1 else model2
    if model:
        v = model.encode(text)
        return v.tolist() if hasattr(v, "tolist") else v
    return None