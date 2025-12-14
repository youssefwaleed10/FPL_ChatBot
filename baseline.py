"""
===========================================================
   FPL KNOWLEDGE GRAPH — BASELINE RETRIEVAL (2.a)
===========================================================
"""

from neo4j import GraphDatabase
import os
import re

# ---------------------------------------------------------
# Config Loading (Robustness Improvement)
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

CONFIG = get_neo4j_config()
URI = CONFIG.get("URI")
AUTH = (CONFIG.get("USERNAME"), CONFIG.get("PASSWORD"))
try:
    # Initialize driver (handling potential None values if config fails)
    driver = GraphDatabase.driver(URI, auth=AUTH) if URI and AUTH[0] else None
    if driver:
        driver.verify_connectivity()
except Exception as e:
    print(f"Failed to connect to Neo4j on startup: {e}")
    driver = None


VALID_SEASONS = ["2021-22", "2022-23"]

def normalize_season(season):
    """Fallback to a valid season if extraction is wrong."""
    if season in VALID_SEASONS:
        return season
    return "2022-23"

# ---------------------------------------------------------
# Cypher Query Templates
# ---------------------------------------------------------

QUERY_TEMPLATES = {

    # 1. QA – Player stats in a specific GW
    "qa_stat_query": """
    MATCH (p:Player {name:$player})-[r:PLAYED_IN]->(f:Fixture {season:$season})
    MATCH (f)<-[:HAS_FIXTURE]-(gw:Gameweek)
    WHERE gw.GW = $gw
    RETURN
      p.name AS player,
      gw.GW AS gameweek,
      r.total_points AS points,
      r.goals_scored AS goals,
      r.assists AS assists,
      r.minutes AS minutes
    """,

    # 2. Top 10 players by total points (season)
    "top_players": """
    MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {season:$season})
    WITH p, SUM(r.total_points) AS total_points
    RETURN
      p.name AS player,
      total_points
    ORDER BY total_points DESC
    LIMIT $limit
    """,

    # 3. Top players by position
    "top_players_by_position": """
    MATCH (p:Player)-[:PLAYS_AS]->(:Position {name:$position})
    MATCH (p)-[r:PLAYED_IN]->(f:Fixture {season:$season})
    WITH p, SUM(r.total_points) AS pts
    RETURN
      p.name AS player,
      pts
    ORDER BY pts DESC
    LIMIT $limit
    """,

    # 4. Compare players (total stats)
    "compare_players": """
    MATCH (p:Player)
    WHERE p.name IN $players
    MATCH (p)-[r:PLAYED_IN]->(f:Fixture {season:$season})
    WITH p,
         SUM(r.total_points) AS points,
         SUM(r.goals_scored) AS goals,
         SUM(r.assists) AS assists
    RETURN
      p.name AS player,
      points,
      goals,
      assists
    ORDER BY points DESC
    """,

    # 5. Similar players based on stats
   "SIMILAR_PLAYERS_QUERY": """
    MATCH (target:Player {name:$player})-[:PLAYS_AS]->(pos:Position)
    MATCH (target)-[tr:PLAYED_IN]->(tf:Fixture {season:$season})
    WITH target, pos,
        SUM(tr.total_points) AS target_points,
        SUM(tr.goals_scored) AS target_goals,
        SUM(tr.assists) AS target_assists

    MATCH (p:Player)-[:PLAYS_AS]->(pos)
    WHERE p <> target
    MATCH (p)-[r:PLAYED_IN]->(f:Fixture {season:$season})
    WITH p,
        target_points, target_goals, target_assists,
        SUM(r.total_points) AS points,
        SUM(r.goals_scored) AS goals,
        SUM(r.assists) AS assists

    WITH p, points, goals, assists,
        ABS(points  - target_points)
    + 10 * ABS(goals - target_goals)
    + 10 * ABS(assists - target_assists) AS similarity_score

    RETURN
    p.name AS similar_player,
    points,
    goals,
    assists,
    similarity_score
    ORDER BY similarity_score ASC
    LIMIT $limit
    """,

    # 6. Fixtures played by a player
    "fixtures_of_player": """
    MATCH (p:Player {name:$player})-[r:PLAYED_IN]->(f:Fixture {season:$season})
    MATCH (f)<-[:HAS_FIXTURE]-(gw:Gameweek)
    RETURN
      p.name AS player,
      gw.GW AS gameweek,
      r.total_points AS points
    ORDER BY gw.GW DESC 
    LIMIT $limit
    """,

    # 7. Differential-style – Best points per 90 (position)
    "points_per_90": """
    MATCH (p:Player)-[:PLAYS_AS]->(:Position {name:$position})
    MATCH (p)-[r:PLAYED_IN]->(f:Fixture {season:$season})
    WHERE r.minutes > 0
    WITH p,
         SUM(r.total_points) AS pts,
         SUM(r.minutes) AS mins
    WHERE mins >= $min_minutes
    WITH p, pts, mins, ROUND(pts * 90.0 / mins, 2) AS pts_per_90
    RETURN
      p.name AS player,
      pts,
      mins,
      pts_per_90
    ORDER BY pts_per_90 DESC, pts DESC
    LIMIT $limit
    """,

   # In baseline.py, inside QUERY_TEMPLATES:
    # # 8. Team analysis – goals conceded (Robust Version)
    # "team_analysis": """
    # MATCH (t:Team {name:$team})
    # MATCH (f:Fixture {season:$season})-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t)
    # MATCH (p:Player)-[r:PLAYED_IN]->(f)
    # WHERE r.team = $team
    # AND r.minutes > 0
    # AND r.goals_conceded IS NOT NULL
    # WITH t, f, MAX(r.goals_conceded) AS conceded_in_fixture
    # RETURN
    # t.name AS team,
    # COUNT(DISTINCT f) AS fixtures_played,
    # SUM(conceded_in_fixture) AS total_goals_conceded
    # """,

    #8. Bonus points leaderboard
    "top_bonus_points": """
    MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {season:$season})
    WITH p,
        SUM(COALESCE(r.bonus, 0)) AS bonus_points,
        SUM(COALESCE(r.total_points, 0)) AS total_points
    RETURN
    p.name AS player,
    bonus_points,
    total_points
    ORDER BY bonus_points DESC, total_points DESC
    LIMIT $limit
    """,


    # 9. Player performances above own season average
    "above_average_performance": """
    MATCH (p:Player {name:$player})-[r:PLAYED_IN]->(f:Fixture {season:$season})
    // 1. Calculate the season average points for the player
    WITH p, AVG(r.total_points) AS avgPts

    // 2. Find all fixtures (fx) where the points (r2.total_points) were greater than the average
    MATCH (p)-[r2:PLAYED_IN]->(fx:Fixture {season:$season})
    MATCH (fx)<-[:HAS_FIXTURE]-(gw:Gameweek)
    WHERE r2.total_points > avgPts

    RETURN
      p.name AS player,
      gw.GW AS gameweek,
      r2.total_points AS points
    ORDER BY points DESC
    """,

    # 10. Upcoming fixtures for a team
    "upcoming_team_fixtures": """
    MATCH (t:Team {name:$team})
    MATCH (f:Fixture {season:$season})-[:HAS_HOME_TEAM|HAS_AWAY_TEAM]->(t)
    MATCH (f)<-[:HAS_FIXTURE]-(gw:Gameweek)
    OPTIONAL MATCH (f)-[:HAS_HOME_TEAM]->(home:Team)
    OPTIONAL MATCH (f)-[:HAS_AWAY_TEAM]->(away:Team)
    WITH t, f, gw, home, away
    WHERE gw.GW >= $current_gw
    RETURN
      t.name AS team,
      gw.GW AS gameweek,
      CASE
        WHEN home.name = t.name THEN away.name
        ELSE home.name
      END AS opponent
    ORDER BY gw.GW ASC
    LIMIT $limit
    """,

    # 12. 🆕 Players with most goal contributions (goals + assists)
    "top_goal_contributors": """
    MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture {season:$season})
    WITH p,
         SUM(r.goals_scored + r.assists) AS contributions,
         SUM(r.total_points) AS points
    RETURN
      p.name AS player,
      contributions,
      points
    ORDER BY contributions DESC
    LIMIT $limit
    """
}


# ---------------------------------------------------------
# Route intent → template (This is the logic we finalized)
# ---------------------------------------------------------

def choose_query(intent, entities):

    # ----------------------------
    # Robust default parameters
    # ----------------------------
    # params = {
    #     "season": normalize_season(entities.get("season")),
    #     "gw": entities.get("gameweek", 5),
    #     "player": entities["players"][0] if entities.get("players") else "Mohamed Salah",
    #     "players": entities.get("players", ["Erling Haaland", "Mohamed Salah"]), # Explicit list for compare
    #     "position": entities.get("position", "MID"),
    #     "team": entities["teams"][0] if entities.get("teams") else "Liverpool",
    #     "threshold": 10.0,
    #     "current_gw": entities.get("gameweek", 5), # Use GW for current_gw if present
    #    "limit": entities.get("limit", 10), # IMPORTANT: Use extracted limit
    #     "min_minutes": 500 # Default for points_per_90
    # }
    params = {
        "season": normalize_season(entities.get("season")),
        "gw": entities.get("gameweek"),
        "player": entities["players"][0] if entities.get("players") else None,
        "players": entities.get("players", []),
        "position": entities.get("position"),   # keep None if not mentioned
        "team": entities["teams"][0] if entities.get("teams") else None,
        "limit": entities.get("limit") or 10,
        "min_minutes": 500,
        "current_gw": entities.get("gameweek")
    }


    # Helper: Get attributes and misc keywords for routing
    attributes = entities.get("attributes", [])
    # Ensure misc_keywords is retrieved, defaulting to empty string if not present
    misc = entities.get("misc_keywords", "").lower() 

    
    # =================================================
    # 2️⃣ INTENT-DRIVEN ROUTING
    # =================================================

    

    if intent == "qa_stat_query":
        return QUERY_TEMPLATES["qa_stat_query"], params

    if intent == "points_per_90":
        return QUERY_TEMPLATES["points_per_90"], params

    if intent == "top_players_by_position":
        return QUERY_TEMPLATES["top_players_by_position"], params

    if intent == "compare_players":
        if len(params["players"]) < 2:
             params["players"] = ["Erling Haaland", "Mohamed Salah"]
        return QUERY_TEMPLATES["compare_players"], params

    if intent == "similar_players":
        return QUERY_TEMPLATES["SIMILAR_PLAYERS_QUERY"], params

    if intent == "fixtures_of_player":
        return QUERY_TEMPLATES["fixtures_of_player"], params

    if intent == "team_analysis":
        return QUERY_TEMPLATES["team_analysis"], params

    if intent == "top_bonus_points":
        return QUERY_TEMPLATES["top_bonus_points"], params

    if intent == "above_average_performance":
        return QUERY_TEMPLATES["above_average_performance"], params

    if intent == "upcoming_team_fixtures":
        return QUERY_TEMPLATES["upcoming_team_fixtures"], params

    if intent == "top_goal_contributors":
        return QUERY_TEMPLATES["top_goal_contributors"], params

    if intent == "top_players":
        # Extra check: if position is MID/DEF/FWD, route to positional query
        if params["position"] and params["position"] != "ALL":
            return QUERY_TEMPLATES["top_players_by_position"], params
        return QUERY_TEMPLATES["top_players"], params
        




    # =================================================
    # 1️⃣ ATTRIBUTE-DRIVEN ROUTING (MOST SPECIFIC)
    # =================================================

    # --- Player form / Recent fixtures ---
    if params["player"] and ("form" in attributes or "fixtures" in attributes or "last" in misc):
        return QUERY_TEMPLATES["fixtures_of_player"], params
    
    
    # --- Team fixtures ---
    # if params["team"] and ("fixtures" in attributes or "upcoming" in misc or "next" in misc):
    #     return QUERY_TEMPLATES["upcoming_team_fixtures"], params
    if params["team"] and ("upcoming" in misc or "next" in misc or "future" in misc or "starting from gameweek" in misc):
        return QUERY_TEMPLATES["upcoming_team_fixtures"], params

    # --- Differential / Cheap / Value players ---
    if "differential" in attributes or "cheap" in attributes or "per 90" in misc:
        return QUERY_TEMPLATES["points_per_90"], params

    # --- Similarity Search ---
    if "similar" in attributes or "similar" in misc:
        return QUERY_TEMPLATES["SIMILAR_PLAYERS_QUERY"], params

    # --- Goal Contributions ---
    if "contribution" in attributes or "contributions" in misc or "goals and assists" in misc:
        return QUERY_TEMPLATES["top_goal_contributors"], params

    # --- Above Average Performance ---
    if "average" in attributes or "average" in misc:
        return QUERY_TEMPLATES["above_average_performance"], params
    
    if "differential" in attributes or "cheap" in attributes or "per 90" in entities.get("misc_keywords", "").lower():
        return QUERY_TEMPLATES["points_per_90"], params





    # =================================================
    # 3️⃣ SAFE FALLBACKS (NEVER FAIL)
    # =================================================

    if params["position"]:
        return QUERY_TEMPLATES["top_players_by_position"], params

    # Default fallback
    return QUERY_TEMPLATES["top_players"], params

# ---------------------------------------------------------
# Run query
# ---------------------------------------------------------

def run_query(cypher, params):
    global driver
    if not driver:
        return [{"Error": "Database connection failed.", "Cypher": cypher, "Params": params}]
    try:
        with driver.session() as session:
            data = [r.data() for r in session.run(cypher, params)]
        return data
    except Exception as e:
        print(f"Error executing Cypher query: {e}")
        return [{"Error": str(e), "Cypher": cypher, "Params": params}]


# ---------------------------------------------------------
# Public function (THE REQUIRED ENTRY POINT)
# ---------------------------------------------------------

def retrieve_from_kg(intent, entities):
    """
    Public function that connects the NLU layer to the database execution layer.
    This function signature is what your llm_layer.py expects.
    """
    # 1. Select Query and Parameters
    cypher, params = choose_query(intent, entities)
    
    if not cypher:
        # Should not happen with the robust logic, but for safety:
        return [{"message": "Could not determine query."}, None] 
    
    # 2. Execute Query
    results = run_query(cypher, params)
    
    # 3. Return results and the Cypher query string
    return results, cypher