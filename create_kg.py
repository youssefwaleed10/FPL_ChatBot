import pandas as pd
from neo4j import GraphDatabase
import os
import re

# ---------------------------------------------------------
# Config Loading (Robustness Improvement)
# ---------------------------------------------------------
def get_neo4j_credentials(config_file="config.txt"):
    config = {}
    # Read config path relative to the current script
    config_path = os.path.join(os.path.dirname(__file__), config_file)
    try:
        with open(config_path, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    config[key] = value
        return config['URI'], config['USERNAME'], config['PASSWORD']
    except Exception as e:
        print(f"Error loading config.txt: {e}")
        return None, None, None

# ---------------------------------------------------------
# 1. Constraints (Fixed property names)
# ---------------------------------------------------------
def create_constraints(tx):
    """Creates unique constraints for FPL schema."""
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Season) REQUIRE s.season IS UNIQUE") # FIX: s.season
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Team) REQUIRE t.name IS UNIQUE") 
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (pos:Position) REQUIRE pos.name IS UNIQUE") 
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Player) REQUIRE p.element IS UNIQUE") # element is safer than (name, element)
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (gw:Gameweek) REQUIRE (gw.season, gw.GW) IS UNIQUE")
    # Fixture constraint simplified: season + fixture ID is usually unique
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (f:Fixture) REQUIRE (f.season, f.fixture) IS UNIQUE")

# ---------------------------------------------------------
# 2. Main Graph Builder (Fixed property names)
# ---------------------------------------------------------
def merge_data_row(tx, row):
    """Creates/Merges all nodes and relationships for a single player's fixture data."""
    
    # Clean data: Neo4j does not like numpy int64/float64, convert to Python standard types
    clean_row = {k: int(v) if isinstance(v, (int, float)) and v == int(v) else v for k, v in row.items()}

    cypher_query = """
    // 1. MERGE all Nodes based on unique identifiers
    MERGE (s:Season {season: $season}) // FIX: s.season property
    
    MERGE (h:Team {name: $home_team})
    MERGE (a:Team {name: $away_team})
    
    MERGE (pos:Position {name: $position})

    // Player (Add team property for consistency, assumes 'team' is in your CSV or must be inferred)
    MERGE (p:Player {element: $element}) // Use element for unique ID
    ON CREATE SET p.name = $name, 
                  p.now_cost = $now_cost, 
                  p.selected_by_percent = $selected_by_percent // Added player properties
    
    MERGE (gw:Gameweek {season: $season, GW: $GW})
    
    MERGE (f:Fixture {season: $season, fixture: $fixture})
    ON CREATE SET f.kickoff_time = $kickoff_time,
                  f.opponent = $opponent // Opponent is useful for a recommendation
    
    // 2. MERGE Structural Relationships
    MERGE (s)-[:HAS_GW]->(gw)
    MERGE (gw)-[:HAS_FIXTURE]->(f)
    MERGE (p)-[:PLAYS_AS]->(pos)
    
    MERGE (f)-[:HAS_HOME_TEAM]->(h)
    MERGE (f)-[:HAS_AWAY_TEAM]->(a)
    
    // FIX: Add PLAYS_FOR relationship based on CSV if possible. If not, this is a current limitation.
    // Assuming you can map a player to a team name in the CSV (e.g., 'team_name' column)
    // MERGE (p)-[:PLAYS_FOR]->(t:Team {name: $team_name}) 
    
    // 3. MERGE the critical PLAYED_IN relationship and set all performance metrics
    MERGE (p)-[playedin:PLAYED_IN]->(f)
    SET playedin.minutes = $minutes,
        playedin.goals_scored = $goals_scored,
        playedin.assists = $assists,
        playedin.total_points = $total_points,
        playedin.bonus = $bonus,
        playedin.clean_sheets = $clean_sheets,
        playedin.goals_conceded = $goals_conceded,
        playedin.own_goals = $own_goals,
        playedin.penalties_saved = $penalties_saved,
        playedin.penalties_missed = $penalties_missed,
        playedin.yellow_cards = $yellow_cards,
        playedin.red_cards = $red_cards,
        playedin.saves = $saves,
        playedin.bps = $bps,
        playedin.influence = $influence,
        playedin.creativity = $creativity,
        playedin.threat = $threat,
        playedin.ict_index = $ict_index,
        playedin.form = $form
    """
    tx.run(cypher_query, **clean_row)

def connect_and_build_graph(config_file):
    URI, USERNAME, PASSWORD = get_neo4j_credentials(config_file)

    if not all([URI, USERNAME, PASSWORD]):
        print("Failed to load Neo4j credentials.")
        return

    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
    
    try:
        driver.verify_connectivity()
        print("Neo4j connection verified. Starting graph creation...")
    except Exception as e:
        print(f"Connection failed: {e}")
        driver.close()
        return

    # Assuming 'fpl_two_seasons.csv' is in the same directory as create_kg.py
    csv_path = os.path.join(os.path.dirname(__file__), 'fpl_two_seasons.csv')
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        driver.close()
        return

    df_players = pd.read_csv(csv_path)
    # Ensure data types are correct for properties that will be indexed
    int_cols = ['element', 'GW', 'fixture', 'total_points', 'now_cost', 'selected_by_percent']
    for col in int_cols:
        if col in df_players.columns:
            df_players[col] = pd.to_numeric(df_players[col], errors='coerce').fillna(0).astype(int)
    
    with driver.session() as session:
        session.execute_write(lambda tx: tx.run("MATCH (n) DETACH DELETE n")) 
        
        session.execute_write(create_constraints) 
        print("Unique constraints created successfully.")
        
        records = df_players.to_dict('records')
        print(f"Starting data insertion of {len(records)} records into Neo4j...")
        
        # Using a single write transaction for efficiency
        def batch_insert(tx, batch):
            for record in batch:
                merge_data_row(tx, record)
        
        BATCH_SIZE = 5000 # Optimized batch size
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i:i + BATCH_SIZE]
            session.execute_write(batch_insert, batch)
            print(f"  -> {i + len(batch)} records processed...")

        print("Graph creation complete.")

    driver.close()
    print("Neo4j connection closed.")

if __name__ == "__main__":
    connect_and_build_graph("config.txt")