#!/usr/bin/env python3
"""
Clash Royale Deck Optimizer using GA
"""
import os
import json
import random
import argparse
import requests
import numpy as np
import xgboost as xgb
from collections import defaultdict

# Config
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("TOKEN", "")
BOOSTED_CARDS_LIST = [c.strip().lower() for c in os.getenv("BOOSTED_CARDS", "").split(",") if c.strip()]
BASE_URL = "https://proxy.royaleapi.dev/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

DATA_DIR = "data"
VECTORIZED_DIR = os.path.join(DATA_DIR, "vectorized")
CARD_DATA_FILE = os.path.join(DATA_DIR, "cardData.json")
CSV_FILE = os.path.join(DATA_DIR, "export_battles.csv")
MODEL_FILE = "xgboost_model.json"
FEATURE_MAP_FILE = os.path.join(VECTORIZED_DIR, "featureIndexMap.json")
META_FILE = os.path.join(VECTORIZED_DIR, "vectorizationMetadata.json")

# Genetic Algorithm Parameters
# Larger population and more generations for better exploration
POPULATION_SIZE = 2000  # Increased for better coverage of search space
ELITE_SIZE = int(POPULATION_SIZE * 0.10)
GENERATIONS = 300  # Increased for convergence from random start
MUTATION_RATE = 0.4  # Relatively low for stability
EARLY_STOP_STREAK = 50
SCORE_STABLE_EPS = 1e-6

# Game Rules
DECK_SIZE = 8
MAX_EVOLUTIONS = 2
MAX_HEROES = 2  # heroes (incl. champions counted as hero-slot cards)
MAX_COMBINED_SPECIALS = 3  # max evo + hero/champion cards combined

def load_card_data():
    with open(CARD_DATA_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data



def get_base_card_id(card_token):
    """Strip _EVO / _HERO suffix from card token to get base card ID."""
    if not card_token:
        return ""
    token_str = str(card_token)
    if token_str.endswith('_EVO'):
        return token_str[:-4]
    if token_str.endswith('_HERO'):
        return token_str[:-5]
    return token_str

def calculate_deck_elixir(deck_ids, card_data):
    id_to_name = card_data.get('id_to_name', {})
    elixir_costs = card_data.get('elixir_costs', {})
    
    total_elixir = 0
    card_count = 0
    
    for card_id in deck_ids:
        base_id = get_base_card_id(card_id)
        card_name = id_to_name.get(base_id, "")
        cost = elixir_costs.get(card_name, 0)
        if cost > 0:  # Don't count tower troops (0 elixir)
            total_elixir += cost
            card_count += 1
    
    return total_elixir / card_count if card_count > 0 else 0

def parse_elixir_range(range_str):
    if not range_str:
        return None, None
    
    range_str = range_str.strip()
    
    if '-' not in range_str:
        # Single value exact (treat as min-max with tolerance)
        try:
            val = float(range_str)
            return val, val
        except ValueError:
            return None, None
    
    parts = range_str.split('-', 1)
    
    min_val = None
    max_val = None
    
    # Parse minimum
    if parts[0].strip():
        try:
            min_val = float(parts[0].strip())
        except ValueError:
            pass
    
    # Parse maximum
    if len(parts) > 1 and parts[1].strip():
        try:
            max_val = float(parts[1].strip())
        except ValueError:
            pass
    
    return min_val, max_val

def check_elixir_constraint(deck_ids, card_data, min_elixir, max_elixir):
    if min_elixir is None and max_elixir is None:
        return True
    
    avg_elixir = calculate_deck_elixir(deck_ids, card_data)
    
    if min_elixir is not None and avg_elixir < min_elixir:
        return False
    
    if max_elixir is not None and avg_elixir >= max_elixir:
        return False
    
    return True

def normalize_level(level, max_level): # Convert legacy to normalized; API still uses 'rares' as starting on level 4, for example.
    return level + (16 - max_level)

def fetch_player_data(player_tag):
    # Ensure tag starts with # for api input
    if not player_tag.startswith('#'):
        player_tag = '#' + player_tag
    
    url = f"{BASE_URL}/players/{requests.utils.quote(player_tag)}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API Error: Status {response.status_code}, Response: {response.text[:200]}")
    except Exception as e:
        print(f"Error fetching player data: {e}")
    return None

def fetch_recent_trophy_road_battle(player_tag):
    if not player_tag.startswith('#'):
        player_tag = '#' + player_tag

    url = f"{BASE_URL}/players/{requests.utils.quote(player_tag)}/battlelog"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            print(f"API Error (battlelog): Status {response.status_code}, Response: {response.text[:200]}")
            return None
        battles = response.json()
    except Exception as e:
        print(f"Error fetching battlelog: {e}")
        return None

    for battle in battles:
        if battle.get('type') != 'PvP':
            continue

        game_mode_name = (battle.get('gameMode') or {}).get('name')
        if game_mode_name != 'Ladder':
            continue

        team = battle.get('team') or []
        opponent = battle.get('opponent') or []
        if len(team) != 1 or len(opponent) != 1:
            continue  # Ensure 1v1 only

        return battle

    return None

def extract_player_inventory(player_data):
    """
    Extract player's available cards with proper level conversion
    Returns: tuple (inventory dict {card_id: normalized_level}, evo_cards set, hero_cards set)
    
    Note: evolutionLevel = 1 means EVO unlocked
          evolutionLevel = 2 means HERO unlocked
          evolutionLevel = 3 means BOTH EVO and HERO unlocked
    Separate upgrade paths, not cumulative.
    """
    inventory = {}
    evo_cards = set()  # Cards with EVO unlocked (evolutionLevel in (1, 3))
    hero_cards = set()  # Cards with HERO unlocked (evolutionLevel in (2, 3))
    
    if not player_data:
        return inventory, evo_cards, hero_cards
    
    cards = player_data.get('cards', [])
    support_cards = player_data.get('supportCards', [])

    # Estimate King Tower Level from Support Cards/Tower Troops. If tower troops are underleveled, will be lower
    # Used for season boost
    inferred_king_level = 1
    if support_cards:
        inferred_king_level = max([normalize_level(c['level'], c['maxLevel']) for c in support_cards])
        print(f"Inferred King Tower Level from API: {inferred_king_level}")

    for card in cards + support_cards:
        card_id = str(card['id'])
        level = card['level']
        max_level = card['maxLevel']
        
        # Convert to normalized level (1-16 scale)
        normalized_level = normalize_level(level, max_level)
        inventory[card_id] = normalized_level
        
        # Track evolution states
        evo_level = card.get('evolutionLevel', 0)
        if evo_level in (1, 3):
            evo_cards.add(card_id)
        if evo_level in (2, 3):
            hero_cards.add(card_id)

    # Apply season boosts
    if BOOSTED_CARDS_LIST:
        try:
            card_data = load_card_data()
            name_to_id = {name.lower(): cid for cid, name in card_data.get('id_to_name', {}).items()}
            boosted_applied = []
            for b_name in BOOSTED_CARDS_LIST:
                matched_id = None
                if b_name in name_to_id:
                    matched_id = name_to_id[b_name]
                else:
                    for full_name, cid in name_to_id.items():
                        if b_name in full_name:
                            matched_id = cid
                            break
                            
                if matched_id:
                    current_level = inventory.get(matched_id, 0)
                    if current_level < inferred_king_level:
                        inventory[matched_id] = inferred_king_level
                        boosted_applied.append(card_data['id_to_name'].get(matched_id, b_name))
            
            if boosted_applied:
                print(f"Applied Season Boosts (Level {inferred_king_level}) to: {', '.join(boosted_applied)}")
        except Exception as e:
            print(f"Error applying season boosts: {e}")
            
    return inventory, evo_cards, hero_cards

def _type_from_id_prefix(card_id):
    """Fallback when cardData lacks an explicit types map (older files)."""
    cid = str(card_id)
    if cid.startswith("159"):
        return "tower_troop"
    if cid.startswith("270"):
        return "building"
    if cid.startswith("280"):
        return "spell"
    return "troop"


def identify_card_types(card_data):
    """
    Categorize cards from cardData (populated by the API crawler), not hand-maintained lists.

    Sources:
      - types[card_id]: troop | building | spell | tower_troop  (ID namespace / API)
      - rarities[name] == 'champion'
      - can_evolve / max_evolution_level for evolution-capable cards

    Returns dict with sets: buildings, support_troops, champions, evolutions, spells, troops, all_cards
    """
    categories = {
        'buildings': set(),
        'support_troops': set(),
        'champions': set(),
        'evolutions': set(),
        'spells': set(),
        'troops': set(),
        'all_cards': set(),
    }

    id_to_name = card_data.get('id_to_name', {})
    types = card_data.get('types', {})
    rarities = card_data.get('rarities', {})
    can_evolve = set(str(c) for c in card_data.get('can_evolve', []))
    max_evo = {str(k): v for k, v in card_data.get('max_evolution_level', {}).items()}

    for card_id, name in id_to_name.items():
        card_id = str(card_id)
        categories['all_cards'].add(card_id)

        card_type = types.get(card_id) or _type_from_id_prefix(card_id)
        if card_type == 'building':
            categories['buildings'].add(card_id)
        elif card_type == 'tower_troop':
            categories['support_troops'].add(card_id)
        elif card_type == 'spell':
            categories['spells'].add(card_id)
        else:
            categories['troops'].add(card_id)

        if rarities.get(name) == 'champion':
            categories['champions'].add(card_id)

        if card_id in can_evolve or int(max_evo.get(card_id, 0) or 0) >= 1:
            categories['evolutions'].add(card_id)

    return categories

def extract_deck_from_battle(battle):
    """Extract deck card IDs, normalized levels, and upgrade tags from a battle entry"""
    team = battle.get('team') or []
    if not team:
        return None

    player_entry = team[0]
    cards = player_entry.get('cards') or []
    if len(cards) != DECK_SIZE:
        return None

    deck_ids = []
    deck_levels = {}
    evo_upgrades = set()
    hero_upgrades = set()

    for card in cards:
        if 'id' not in card or 'level' not in card or 'maxLevel' not in card:
            return None

        card_id = str(card['id'])
        normalized_level = normalize_level(card['level'], card['maxLevel'])

        deck_ids.append(card_id)
        deck_levels[card_id] = normalized_level

        evo_level = card.get('evolutionLevel', 0)
        if evo_level == 1:
            evo_upgrades.add(card_id)
        elif evo_level == 2:
            hero_upgrades.add(card_id)

    trophy_snapshot = player_entry.get('startingTrophies') or player_entry.get('trophyCount')

    # Extract support cards (Tower Troops)
    support_cards = player_entry.get('supportCards') or []
    tower_troop_id = None
    if support_cards:
        tc = support_cards[0]
        tower_troop_id = str(tc['id'])
        normalized_level = normalize_level(tc['level'], tc['maxLevel'])
        deck_levels[tower_troop_id] = normalized_level

    return deck_ids, deck_levels, evo_upgrades, hero_upgrades, trophy_snapshot, tower_troop_id

def build_gauntlet(player_trophies, csv_file, trophy_range=300, gauntlet_size=500):
    """
    Module 1: Create meta gauntlet of enemy decks from CSV data
    Returns: list of dicts with 'loadout' and 'weight'
    """
    import csv
    
    gauntlet = []
    deck_frequency = defaultdict(int)
    card_levels_sum = defaultdict(lambda: defaultdict(int))  # deck_sig -> {card_id: sum of levels}
    card_levels_count = defaultdict(lambda: defaultdict(int))  # deck_sig -> {card_id: count}
    
    min_trophies = player_trophies - trophy_range
    max_trophies = player_trophies + trophy_range
    
    print(f"Building gauntlet for trophy range {min_trophies}-{max_trophies}...")
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                opp_trophies = int(row['opponent_trophies'])
                if min_trophies <= opp_trophies <= max_trophies:
                    loadout_str = row['opponent_loadout']
                    loadout = json.loads(loadout_str)
                    
                    # Extract card IDs (keep _EVO and _HERO suffixes for signature)
                    card_ids = {}
                    for card_key, level in loadout.items():
                        card_ids[card_key] = level
                    
                    # Use sorted card IDs as deck signature
                    deck_sig = tuple(sorted(card_ids.keys()))
                    deck_frequency[deck_sig] += 1
                    
                    # Track levels for averaging
                    for card_id, level in card_ids.items():
                        card_levels_sum[deck_sig][card_id] += level
                        card_levels_count[deck_sig][card_id] += 1
                        
            except (ValueError, json.JSONDecodeError, KeyError):
                continue
    
    # Convert to weighted list (limit to top gauntlet_size for better meta coverage)
    sorted_decks = sorted(deck_frequency.items(), key=lambda x: x[1], reverse=True)[:gauntlet_size]
    
    total_count = sum(count for _, count in sorted_decks)
    
    for deck_sig, count in sorted_decks:
        # Calculate average levels for each card in this deck
        loadout = {}
        for card_id in deck_sig:
            avg_level = card_levels_sum[deck_sig][card_id] / card_levels_count[deck_sig][card_id]
            loadout[card_id] = round(avg_level)  # Round to nearest integer level
        
        weight = count / total_count if total_count > 0 else 0
        
        gauntlet.append({
            'loadout': loadout,
            'weight': weight
        })
    
    print(f"Gauntlet built with {len(gauntlet)} unique decks")
    return gauntlet

def get_available_variants_for_card(base_id, evo_cards, hero_cards, card_categories):
    """Return list of possible variant tokens for a base card ID."""
    base_id = get_base_card_id(base_id)
    if card_categories and base_id in card_categories.get('champions', set()):
        return [base_id]  # Champions are always champions
    variants = [base_id]  # Regular is always an option
    if evo_cards and base_id in evo_cards:
        variants.append(f"{base_id}_EVO")
    if hero_cards and base_id in hero_cards:
        variants.append(f"{base_id}_HERO")
    return variants

def is_valid_deck(individual, inventory, card_categories, evo_cards=None, hero_cards=None, required_card_ids=None, card_data=None, min_elixir=None, max_elixir=None):
    """
    Module 2: Validate deck composition against game rules.
    individual is tuple (deck_cards, tower_troop_id).
    deck_cards is a list of card tokens (e.g. '26000000', '26000000_EVO', '26000000_HERO').
    Enforces:
      - Exactly 8 cards.
      - Exactly 1 variation per card (all 8 base card IDs must be distinct).
      - Max 2 EVO cards.
      - Max 2 HERO / Champion cards.
      - Max 3 EVO + HERO/Champion combined.
      - Player owns the base card in inventory.
      - EVO variant only if base card in evo_cards.
      - HERO variant only if base card in hero_cards (or Champion).
      - Tower troop valid.
      - Required cards present.
      - Elixir constraint.
    """
    if not isinstance(individual, tuple) or len(individual) != 2:
        return False
    
    deck_cards, tower_troop_id = individual

    if len(deck_cards) != DECK_SIZE:
        return False
    
    # Check distinct base cards: cannot have multiple variations of same card in deck
    base_ids = [get_base_card_id(c) for c in deck_cards]
    if len(set(base_ids)) != DECK_SIZE:
        return False

    # Check tower troop
    tower_base_id = get_base_card_id(tower_troop_id)
    if tower_base_id not in inventory:
        return False
    if tower_base_id not in card_categories['support_troops']:
        return False

    # Check all cards are available in inventory and not support troops
    for c, base_id in zip(deck_cards, base_ids):
        if base_id not in inventory:
            return False
        if base_id in card_categories['support_troops']:
            return False

        # Check unlock requirements for variants
        if c.endswith('_EVO'):
            if evo_cards is not None and base_id not in evo_cards:
                return False
        elif c.endswith('_HERO'):
            if hero_cards is not None and base_id not in hero_cards and base_id not in card_categories['champions']:
                return False

    # Check evo / hero limits (up to 2 of each, 3 combined; champions count as hero-slot)
    evo_count = sum(1 for c in deck_cards if c.endswith('_EVO'))
    if evo_count > MAX_EVOLUTIONS:
        return False

    hero_count = sum(1 for c, base_id in zip(deck_cards, base_ids) if c.endswith('_HERO') or base_id in card_categories['champions'])
    if hero_count > MAX_HEROES:
        return False
            
    if (evo_count + hero_count) > MAX_COMBINED_SPECIALS:
        return False

    # Enforce required card when specified
    if required_card_ids:
        reqs = [required_card_ids] if isinstance(required_card_ids, str) else required_card_ids
        for req in reqs:
            req_base = get_base_card_id(req)
            if req.endswith('_EVO') or req.endswith('_HERO'):
                if req not in deck_cards:
                    return False
            else:
                if req_base not in base_ids:
                    return False
    
    # Check elixir constraint
    if card_data is not None and (min_elixir is not None or max_elixir is not None):
        if not check_elixir_constraint(deck_cards, card_data, min_elixir, max_elixir):
            return False
    
    return True

def generate_random_deck(inventory, card_categories, evo_cards, hero_cards, required_card_ids=None, excluded_card_ids=None, card_data=None, min_elixir=None, max_elixir=None):
    """
    Generate a random valid deck.
    8 unique base cards with assigned variants (regular, evo, or hero).
    Required cards are forcibly included when provided.
    """
    excluded_base_ids = set()
    excluded_variants = set()
    if excluded_card_ids:
        for x in excluded_card_ids:
            if x.endswith('_EVO') or x.endswith('_HERO'):
                excluded_variants.add(x)
            else:
                excluded_base_ids.add(get_base_card_id(x))

    available_base_cards = [c for c in inventory.keys() if c not in card_categories['support_troops'] and c not in excluded_base_ids]

    # Select Tower Troop (Support Troop)
    support_options = [c for c in inventory.keys() if c in card_categories['support_troops']]
    if not support_options:
        tower_troop_id = "159000000"
    else:
        tower_troop_id = weighted_random_card(support_options, inventory)

    # Handle required cards
    deck_cards = []
    deck_base_ids = []
    
    if required_card_ids:
        reqs = [required_card_ids] if isinstance(required_card_ids, str) else required_card_ids
        for req in reqs:
            req_base = get_base_card_id(req)
            if req_base not in inventory or req_base in card_categories['support_troops']:
                return ([], tower_troop_id)
            if req_base in deck_base_ids:
                continue
            deck_base_ids.append(req_base)
            if req.endswith('_EVO') or req.endswith('_HERO'):
                deck_cards.append(req)
            else:
                deck_cards.append(req_base)

    # Pick remaining base cards
    while len(deck_base_ids) < DECK_SIZE:
        candidates = [c for c in available_base_cards if c not in deck_base_ids]
        if not candidates:
            break
        pick = weighted_random_card(candidates, inventory)
        deck_base_ids.append(pick)
        deck_cards.append(pick)

    if len(deck_base_ids) < DECK_SIZE:
        return ([], tower_troop_id)

    # Assign variants (regular, evo, hero) respecting constraints
    current_evos = sum(1 for c in deck_cards if c.endswith('_EVO'))
    current_heroes = sum(1 for c in deck_cards if c.endswith('_HERO') or get_base_card_id(c) in card_categories['champions'])

    indices = list(range(len(deck_cards)))
    random.shuffle(indices)

    for idx in indices:
        current_token = deck_cards[idx]
        base_id = get_base_card_id(current_token)
        
        # If already assigned a variant or champion, keep it
        if current_token.endswith('_EVO') or current_token.endswith('_HERO') or base_id in card_categories['champions']:
            continue
            
        allowed = [base_id]
        can_evo = (base_id in evo_cards) and (current_evos < MAX_EVOLUTIONS) and ((current_evos + current_heroes) < MAX_COMBINED_SPECIALS) and (f"{base_id}_EVO" not in excluded_variants)
        can_hero = (base_id in hero_cards) and (current_heroes < MAX_HEROES) and ((current_evos + current_heroes) < MAX_COMBINED_SPECIALS) and (f"{base_id}_HERO" not in excluded_variants)

        if can_evo:
            allowed.append(f"{base_id}_EVO")
        if can_hero:
            allowed.append(f"{base_id}_HERO")

        if len(allowed) > 1 and random.random() < 0.7:
            choice = random.choice([v for v in allowed if v != base_id])
        else:
            choice = base_id

        deck_cards[idx] = choice
        if choice.endswith('_EVO'):
            current_evos += 1
        elif choice.endswith('_HERO'):
            current_heroes += 1

    return (deck_cards, tower_troop_id)

def weighted_random_card(candidates, inventory, power=2):
    """
    Select a random card from candidates, weighted by card level.
    Higher level cards are more likely to be chosen.
    """
    if not candidates:
        return None
        
    weights = []
    for c in candidates:
        base_id = get_base_card_id(c)
        level = inventory.get(base_id, inventory.get(c, 1))
        weights.append(level ** power)
    
    return random.choices(candidates, weights=weights, k=1)[0]

class DeckEvaluator:
    """
    Helper class for efficient batch evaluation of decks against the gauntlet.
    Handles vectorization, batch prediction, and caching.
    """
    def __init__(self, gauntlet, model, feature_map, n_features, player_trophies, training_max_trophies, evo_cards=None, hero_cards=None):
        self.gauntlet = gauntlet
        self.model = model
        self.feature_map = feature_map
        self.n_features = n_features
        self.player_trophies = player_trophies
        self.training_max_trophies = max(training_max_trophies, 1)
        self.evo_cards = evo_cards or set()
        self.hero_cards = hero_cards or set()
        self.cache = {}  # signature -> score

        # Precompute opponent features for the gauntlet
        # We store the opponent part of the feature vector + metadata
        self.gauntlet_vectors = np.zeros((len(gauntlet), n_features + 2), dtype=np.float32)
        self.gauntlet_weights = np.array([g['weight'] for g in gauntlet], dtype=np.float32)
        
        for i, enemy in enumerate(gauntlet):
            enemy_loadout = enemy['loadout']
            # Opponent cards
            o_total_level = 0
            o_count = 0
            for card_id, level in enemy_loadout.items():
                o_total_level += level
                o_count += 1
                if card_id in feature_map:
                    idx = feature_map[card_id]
                    self.gauntlet_vectors[i, idx] = level / 16.0
            
            # Opponent metadata
            # We store: [Opponent Trophies normalized], [Opponent Avg Level]
            # Assume opponent has same trophies as player for fairness in matchmaking simulation
            self.gauntlet_vectors[i, n_features] = player_trophies / self.training_max_trophies
            self.gauntlet_vectors[i, n_features + 1] = (o_total_level / o_count / 16.0) if o_count > 0 else 0

    def evaluate(self, population, inventory):
        """
        Evaluate a list of decks (population) against the gauntlet.
        Returns a list of scores corresponding to the population.
        """
        scores = [0.0] * len(population)
        signatures = []
        new_indices = [] # Indices in population that need evaluation
        
        # 1. Check cache
        for i, individual in enumerate(population):
            if isinstance(individual, tuple):
                deck, tower = individual
                sig = tuple(sorted(deck + [tower]))
            else:
                sig = tuple(sorted(individual))
                
            signatures.append(sig)
            if sig in self.cache:
                scores[i] = self.cache[sig]
            else:
                new_indices.append(i)
        
        if not new_indices:
            return scores

        # 2. Prepare batch input
        num_new = len(new_indices)
        num_gauntlet = len(self.gauntlet)
        total_rows = num_new * num_gauntlet
        
        # Full vector size = (n_features * 2) + 5
        X = np.zeros((total_rows, (self.n_features * 2) + 5), dtype=np.float32)
        
        p_trophies_norm = self.player_trophies / self.training_max_trophies
        
        # Construct vectors
        for k, pop_idx in enumerate(new_indices):
            individual = population[pop_idx]
            if isinstance(individual, tuple):
                deck, tower = individual
                full_deck = deck + [tower]
            else:
                full_deck = individual
            
            # Compute player vector part (once per deck)
            p_vector = np.zeros(self.n_features + 1, dtype=np.float32) # cards + avg_level
            p_total_level = 0
            p_count = 0
            for card_item in full_deck:
                base_id = get_base_card_id(card_item)
                level = inventory.get(base_id, inventory.get(card_item, 0)) # Should exist
                p_total_level += level
                p_count += 1
                
                # Determine feature ID based on card_item (or fallback to base_id)
                feature_card_id = card_item
                if feature_card_id not in self.feature_map:
                    feature_card_id = base_id
                    
                if feature_card_id in self.feature_map:
                    idx_feat = self.feature_map[feature_card_id]
                    p_vector[idx_feat] = level / 16.0
            
            p_avg_lvl = (p_total_level / p_count / 16.0) if p_count > 0 else 0
            p_vector[self.n_features] = p_avg_lvl

            # Fill into X for all gauntlet opponents
            start_row = k * num_gauntlet
            end_row = start_row + num_gauntlet
            
            # Player features (0:n_features) -> repeated for all opponents
            X[start_row:end_row, 0:self.n_features] = p_vector[0:self.n_features]
            
            # Opponent features (n_features:2*n_features) -> copied from precomputed gauntlet
            X[start_row:end_row, self.n_features:2*self.n_features] = self.gauntlet_vectors[:, 0:self.n_features]
            
            # Metadata
            # [-5] Player Trophies
            X[start_row:end_row, -5] = p_trophies_norm
            # [-4] Opponent Trophies (from gauntlet)
            X[start_row:end_row, -4] = self.gauntlet_vectors[:, self.n_features]
            # [-3] Player Avg Level
            X[start_row:end_row, -3] = p_vector[self.n_features]
            # [-2] Opponent Avg Level (from gauntlet)
            X[start_row:end_row, -2] = self.gauntlet_vectors[:, self.n_features + 1]
            # [-1] Trophy Diff
            X[start_row:end_row, -1] = p_trophies_norm - self.gauntlet_vectors[:, self.n_features]

        # 3. Batch Predict
        probs = self.model.predict_proba(X)[:, 1] # Probability of winning
        
        # 4. Aggregate scores
        weight_boosts = 1.0 + (self.gauntlet_weights * 2.0)
        total_weight_denom = np.sum(self.gauntlet_weights * weight_boosts)
        total_raw_weight = np.sum(self.gauntlet_weights)

        for k, pop_idx in enumerate(new_indices):
            start_row = k * num_gauntlet
            end_row = start_row + num_gauntlet
            deck_probs = probs[start_row:end_row]
            
            # Weighted average score
            weighted_probs = deck_probs * weight_boosts * self.gauntlet_weights
            avg_score = np.sum(weighted_probs) / total_weight_denom if total_weight_denom > 0 else 0
            
            if total_raw_weight > 0:
                w_mean = np.sum(deck_probs * self.gauntlet_weights) / total_raw_weight
                w_variance = np.sum(self.gauntlet_weights * (deck_probs - w_mean)**2) / total_raw_weight
                variance_penalty = 0.1 * np.sqrt(w_variance)
            else:
                variance_penalty = 0
                
            final_score = max(avg_score - variance_penalty, 0.0)
            
            self.cache[signatures[pop_idx]] = final_score
            scores[pop_idx] = final_score
            
        return scores

def calculate_deck_score(individual, inventory, gauntlet, model, feature_map, n_features, player_trophies, training_max_trophies, evo_cards=None, hero_cards=None):
    """
    Wrapper to calculate deck score using DeckEvaluator (legacy/single use)
    """
    evaluator = DeckEvaluator(gauntlet, model, feature_map, n_features, player_trophies, training_max_trophies, evo_cards, hero_cards)
    scores = evaluator.evaluate([individual], inventory)
    return scores[0]

def crossover(individual1, individual2, inventory, card_categories, evo_cards, hero_cards, excluded_card_ids=None, card_data=None, min_elixir=None, max_elixir=None):
    """
    Module 3: Breed two decks to create offspring
    """
    deck1, tower1 = individual1
    deck2, tower2 = individual2
    
    # Split point
    split = random.randint(3, 5)
    
    child = list(deck1[:split])
    child_base_ids = set(get_base_card_id(c) for c in child)
    
    # Add cards from deck2 that aren't duplicates
    for card in deck2:
        base_id = get_base_card_id(card)
        if base_id not in child_base_ids:
            child.append(card)
            child_base_ids.add(base_id)
        if len(child) >= DECK_SIZE:
            break
    
    # Fill remaining with random cards if needed
    excluded_base_ids = set()
    if excluded_card_ids:
        for x in excluded_card_ids:
            excluded_base_ids.add(get_base_card_id(x))

    available = [c for c in inventory.keys() if c not in card_categories['support_troops'] and c not in child_base_ids and c not in excluded_base_ids]

    while len(child) < DECK_SIZE and available:
        pick = weighted_random_card(available, inventory)
        child.append(pick)
        child_base_ids.add(pick)
        available.remove(pick)
        
    # Reconcile evo / hero slot overages in child
    evo_tokens = [c for c in child if c.endswith('_EVO')]
    while len(evo_tokens) > MAX_EVOLUTIONS:
        downgrade = random.choice(evo_tokens)
        evo_tokens.remove(downgrade)
        idx = child.index(downgrade)
        child[idx] = get_base_card_id(downgrade)
        
    hero_tokens = [c for c in child if c.endswith('_HERO') or get_base_card_id(c) in card_categories['champions']]
    while len(hero_tokens) > MAX_HEROES:
        downgrade_candidates = [c for c in hero_tokens if c.endswith('_HERO')]
        if not downgrade_candidates:
            break
        downgrade = random.choice(downgrade_candidates)
        hero_tokens.remove(downgrade)
        idx = child.index(downgrade)
        child[idx] = get_base_card_id(downgrade)
        
    while (len(evo_tokens) + len(hero_tokens)) > MAX_COMBINED_SPECIALS:
        downgrade_candidates = [c for c in child if c.endswith('_EVO') or c.endswith('_HERO')]
        if not downgrade_candidates:
            break
        downgrade = random.choice(downgrade_candidates)
        if downgrade.endswith('_EVO'):
            evo_tokens.remove(downgrade)
        else:
            hero_tokens.remove(downgrade)
        idx = child.index(downgrade)
        child[idx] = get_base_card_id(downgrade)
        
    child_tower = random.choice([tower1, tower2])
    
    return (child, child_tower)

def mutate(individual, inventory, card_categories, evo_cards, hero_cards, required_card_ids=None, excluded_card_ids=None, card_data=None, min_elixir=None, max_elixir=None):
    """
    Module 3: Randomly mutate deck (card swaps or variant toggles)
    """
    if random.random() > MUTATION_RATE:
        return individual
    
    deck, tower = individual
    deck_copy = list(deck)
    
    req_base_ids = set()
    if required_card_ids:
        reqs = [required_card_ids] if isinstance(required_card_ids, str) else required_card_ids
        for r in reqs:
            req_base_ids.add(get_base_card_id(r))
    
    excluded_base_ids = set()
    if excluded_card_ids:
        for x in excluded_card_ids:
            excluded_base_ids.add(get_base_card_id(x))

    # Mutation: either swap a card or change variant
    if random.random() < 0.5:
        # Variant mutation
        idx = random.randrange(len(deck_copy))
        current_token = deck_copy[idx]
        base_id = get_base_card_id(current_token)
        
        possible_variants = get_available_variants_for_card(base_id, evo_cards, hero_cards, card_categories)
        if len(possible_variants) > 1:
            other_variants = [v for v in possible_variants if v != current_token]
            deck_copy[idx] = random.choice(other_variants)
    else:
        # Card swap mutation
        remove_candidates = [c for c in deck_copy if get_base_card_id(c) not in req_base_ids]
        if remove_candidates:
            to_remove = random.choice(remove_candidates)
            idx = deck_copy.index(to_remove)
            
            current_base_ids = set(get_base_card_id(c) for c in deck_copy)
            available_base = [c for c in inventory.keys() if c not in card_categories['support_troops'] and c not in current_base_ids and c not in excluded_base_ids]
            
            if available_base:
                pick_base = weighted_random_card(available_base, inventory)
                possible_variants = get_available_variants_for_card(pick_base, evo_cards, hero_cards, card_categories)
                deck_copy[idx] = random.choice(possible_variants)
    
    # Mutate tower troop (independent)
    if random.random() < 0.1:
         support_options = [c for c in inventory.keys() if c in card_categories['support_troops'] and c != tower]
         if support_options:
             tower = weighted_random_card(support_options, inventory)

    return (deck_copy, tower)

def genetic_algorithm(inventory, card_categories, gauntlet, model, feature_map, n_features, player_trophies, training_max_trophies, evo_cards, hero_cards, required_card_ids=None, excluded_card_ids=None, card_data=None, min_elixir=None, max_elixir=None):
    """
    Module 3: Run genetic algorithm to optimize deck
    """
    print("\nInitializing population...")
    
    # Initialize Evaluator
    evaluator = DeckEvaluator(gauntlet, model, feature_map, n_features, player_trophies, training_max_trophies, evo_cards, hero_cards)
    
    # Generation 0
    population = []
    for _ in range(POPULATION_SIZE):
        deck = generate_random_deck(inventory, card_categories, evo_cards, hero_cards, required_card_ids, excluded_card_ids, card_data, min_elixir, max_elixir)
        if is_valid_deck(deck, inventory, card_categories, evo_cards, hero_cards, required_card_ids, card_data, min_elixir, max_elixir):
            population.append(deck)
    
    print(f"Generated {len(population)} valid initial decks")
    
    if len(population) < 10:
        print("ERROR: Not enough valid decks in initial population")
        return None, 0.0
    
    best_score = 0.0
    best_deck = None
    stable_streak = 0
    
    generation = 0
    try:
        while True:
            # Evaluate fitness (Batch mode)
            scores = evaluator.evaluate(population, inventory)
            scored_population = list(zip(population, scores))
            
            # Sort by score
            scored_population.sort(key=lambda x: x[1], reverse=True)
            
            # Track best
            current_best_score = scored_population[0][1]
            if current_best_score > best_score + SCORE_STABLE_EPS:
                best_score = current_best_score
                best_deck = scored_population[0][0]
                stable_streak = 0
            elif abs(current_best_score - best_score) <= SCORE_STABLE_EPS:
                stable_streak += 1
            else:
                stable_streak = 0
            
            if generation % 10 == 0:
                print(f"Generation {generation}: Best Score = {best_score:.4f}")

            if stable_streak >= EARLY_STOP_STREAK:
                print(f"Early stop at generation {generation} after {EARLY_STOP_STREAK} steady best scores ({best_score:.4f})")
                break
            
            # If we've reached max generations (safety break if not using while loop condition)
            if generation >= GENERATIONS:
                break

            generation += 1
            
            # Selection: Keep elite
            elite = [deck for deck, score in scored_population[:ELITE_SIZE]]
            
            # Breeding: Create offspring
            new_population = elite.copy()
            while len(new_population) < POPULATION_SIZE:
                # Select two parents from elite
                parent1 = random.choice(elite)
                parent2 = random.choice(elite)
                
                # Crossover
                child = crossover(parent1, parent2, inventory, card_categories, evo_cards, hero_cards, excluded_card_ids, card_data, min_elixir, max_elixir)
                
                # Mutation
                child = mutate(child, inventory, card_categories, evo_cards, hero_cards, required_card_ids, excluded_card_ids, card_data, min_elixir, max_elixir)
                
                # Validate
                if is_valid_deck(child, inventory, card_categories, evo_cards, hero_cards, required_card_ids, card_data, min_elixir, max_elixir):
                    new_population.append(child)
            
            population = new_population
    except KeyboardInterrupt:
        print(f"\nInterrupted at generation {generation}. Returning best deck found so far (score {best_score:.4f}).")
    
    print(f"\nOptimization complete! Final best score: {best_score:.4f}")
    return best_deck, best_score

def calculate_consistency_score(deck_ids, card_data):
    """Calculate deck consistency score (inverse of elixir standard deviation)
    Lower variance = more consistent = higher score
    """
    id_to_name = card_data.get('id_to_name', {})
    elixir_costs = card_data.get('elixir_costs', {})
    
    costs = []
    for card_id in deck_ids:
        card_name = id_to_name.get(card_id, "")
        cost = elixir_costs.get(card_name, 0)
        if cost > 0:  # Don't count tower troops
            costs.append(cost)
    
    if len(costs) == 0:
        return 0
    
    mean = sum(costs) / len(costs)
    variance = sum((x - mean) ** 2 for x in costs) / len(costs)
    std_dev = variance ** 0.5
    
    # Normalize to 0-100 scale (lower std_dev = higher score)
    # Typical std_dev ranges from 0-2.5, so we invert and scale
    max_std_dev = 3.0
    consistency = max(0, (max_std_dev - std_dev) / max_std_dev * 100)
    
    return consistency

def format_deck_output(individual, inventory, card_data, evo_cards=None, hero_cards=None, upgrade_values=None):
    """
    Format final deck output with evos and heroes labeled.
    Guarantees at most 1 variation modifier per card (EVO, HERO, or none).
    """
    if isinstance(individual, tuple):
        deck, tower = individual
    else:
        deck = individual
        tower = None

    id_to_name = card_data.get('id_to_name', {})
    card_categories = identify_card_types(card_data)
    
    output_lines = []
    
    for card_token in deck:
        base_id = get_base_card_id(card_token)
        name = id_to_name.get(base_id, f"Unknown({base_id})")
        level = inventory.get(base_id, 0)
        
        tags = []
        if card_token.endswith('_HERO'):
            tags.append("HERO")
        elif card_token.endswith('_EVO'):
            tags.append("EVO")
        elif base_id in card_categories['champions']:
            tags.append("CHAMPION")
        
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        
        uv_str = ""
        if upgrade_values and (base_id in upgrade_values or card_token in upgrade_values):
            uv = upgrade_values.get(base_id, upgrade_values.get(card_token, 0))
            uv_str = f"     UV: {uv:.1f}%"
            
        output_lines.append(f"  - {name} (Level {level}){tag_str}{uv_str}")
    
    if tower:
        tower_base = get_base_card_id(tower)
        name = id_to_name.get(tower_base, f"Unknown({tower_base})")
        level = inventory.get(tower_base, 0)
        output_lines.append(f"  - {name} (Level {level}) [TOWER TROOP]")

    return "\n".join(output_lines)

def resolve_card_identifier(name_or_id, card_data):
    """
    Resolve a card identifier that may be an ID or a name with optional "evo " or "hero " prefix.
    Returns the card ID string (e.g. "26000000", "26000000_EVO", "26000000_HERO") or None if not found.
    """
    id_to_name = card_data.get('id_to_name', {})
    name_to_id = {name.lower(): cid for cid, name in id_to_name.items()}

    token = name_or_id.strip()
    is_evo = token.lower().startswith('evo ') or token.endswith('_EVO')
    is_hero = token.lower().startswith('hero ') or token.endswith('_HERO')

    clean = token.lower()
    if clean.startswith('evo '):
        clean = clean[4:].strip()
    elif clean.startswith('hero '):
        clean = clean[5:].strip()
    if clean.endswith('_evo'):
        clean = clean[:-4].strip()
    elif clean.endswith('_hero'):
        clean = clean[:-5].strip()

    base_id = None
    if clean in id_to_name:
        base_id = clean
    elif clean in name_to_id:
        base_id = name_to_id[clean]
    else:
        for full_name, cid in name_to_id.items():
            if clean in full_name or full_name in clean:
                base_id = cid
                break

    if not base_id:
        return None

    if is_evo:
        return f"{base_id}_EVO"
    elif is_hero:
        return f"{base_id}_HERO"
    return base_id

def test_specific_deck(deck_names, inventory, card_data, card_categories, gauntlet, model, feature_map, n_features, player_trophies, training_max_trophies, evo_cards, hero_cards):
    """
    Test a specific deck by name
    Returns: (deck_ids, win_rate) or (None, 0) if invalid
    """
    deck_ids = []
    tower_troop_id = "159000000" # Default Tower Princess

    for name in deck_names:
        cid = resolve_card_identifier(name, card_data)
        if not cid:
            print(f"ERROR: Card '{name}' not found")
            return None, 0.0
            
        base_id = get_base_card_id(cid)
        if base_id in card_categories['support_troops']:
            tower_troop_id = base_id
        else:
            deck_ids.append(cid)
    
    individual = (deck_ids, tower_troop_id)

    # Validate deck
    if not is_valid_deck(individual, inventory, card_categories, evo_cards, hero_cards):
        print("ERROR: Invalid deck composition")
        print(f"  - Deck size: {len(deck_ids)}")
        evo_count = sum(1 for c in deck_ids if c.endswith('_EVO'))
        print(f"  - Evo cards: {evo_count} (max: {MAX_EVOLUTIONS})")
        hero_count = sum(1 for c in deck_ids if c.endswith('_HERO') or get_base_card_id(c) in card_categories['champions'])
        print(f"  - Hero/Champion cards: {hero_count} (max: {MAX_HEROES})")
        print(f"  - Combined Specials: {evo_count + hero_count} (max: {MAX_COMBINED_SPECIALS})")
        
        base_ids = [get_base_card_id(c) for c in deck_ids]
        if len(set(base_ids)) != len(deck_ids):
            print(f"  - Duplicate cards detected in deck!")
            
        tower_base = get_base_card_id(tower_troop_id)
        if tower_base not in inventory:
             print(f"  - Tower troop {card_data['id_to_name'].get(tower_base, tower_base)} not in inventory")
        elif tower_base not in card_categories['support_troops']:
             print(f"  - {card_data['id_to_name'].get(tower_base, tower_base)} is not a tower troop")
             
        unavailable = [c for c in base_ids if c not in inventory]
        if unavailable:
            print(f"  - Unavailable cards: {[card_data['id_to_name'].get(c, c) for c in unavailable]}")
        return None, 0.0
    
    # Calculate win rate
    win_rate = calculate_deck_score(individual, inventory, gauntlet, model, feature_map, n_features, player_trophies, training_max_trophies, evo_cards, hero_cards)
    
    return individual, win_rate

def calculate_upgrade_values(deck, evaluator, inventory, base_win_rate):
    """
    Calculate win rate increase for upgrading each card in the deck
    """
    upgrade_values = {}
    
    # Handle tuple deck (deck, tower)
    if isinstance(deck, tuple):
        card_ids = deck[0]
        tower_id = deck[1]
        all_cards = list(card_ids) + ([tower_id] if tower_id else [])
    else:
        all_cards = list(deck)

    for card_token in all_cards:
        base_id = get_base_card_id(card_token)
        current_level = inventory.get(base_id, 0)
        # Only check upgrades for cards under level 16
        if current_level >= 16 or current_level == 0:
            continue
            
        # Temporarily upgrade card
        original_level = inventory[base_id]
        inventory[base_id] = current_level + 1
        
        # Clear cache for this specific evaluation because cache key doesn't include levels
        evaluator.cache = {}
        
        # Evaluate
        scores = evaluator.evaluate([deck], inventory)
        new_win_rate = scores[0]
        
        # Restore level
        inventory[base_id] = original_level
        
        uv = (new_win_rate - base_win_rate)
        if uv > 0.0001: # Filter tiny improvements
             upgrade_values[base_id] = uv * 100
             upgrade_values[card_token] = uv * 100
             
    return upgrade_values

def main():
    parser = argparse.ArgumentParser(description="Clash Royale Deck Optimizer")
    parser.add_argument("--required", type=str, default=None, help="Card ID or name to force include (supports 'evo ' / 'hero ' prefixes)")
    parser.add_argument("--exclude", type=str, default=None, help="Card ID or name to force exclude (supports 'evo ' / 'hero ' prefixes)")
    parser.add_argument("--tag", type=str, default="CLL0LCPPJ", help="Player tag to analyze (default: CLL0LCPPJ)")
    parser.add_argument("--elixir", type=str, default=None, help="Elixir range filter (e.g., '2.5-5.0', '2.5-', or '-5.0')")
    parser.add_argument("--assume-all-evos", action="store_true", help="Treat every evolution-capable card as unlocked (heroes stay heroes)")
    parser.add_argument("--assume-all-max-level", action="store_true", help="Treat every owned card as max level (16)")
    parser.add_argument("--gauntlet-size", type=int, default=500, help="Number of top meta decks to include in the gauntlet (default: 500)")
    args = parser.parse_args()
    
    # Parse elixir range
    min_elixir, max_elixir = parse_elixir_range(args.elixir)
    if args.elixir:
        range_desc = []
        if min_elixir is not None:
            range_desc.append(f">= {min_elixir}")
        if max_elixir is not None:
            range_desc.append(f"< {max_elixir}")
        print(f"\nElixir constraint: {' and '.join(range_desc) if range_desc else 'None'}")

    print("=" * 60)
    print("Clash Royale Deck Optimizer")
    print("=" * 60)
    
    # Load dependencies
    print("\nLoading card data...")
    card_data = load_card_data()
    card_categories = identify_card_types(card_data)

    required_card_ids = []
    if args.required:
        tokens = [t.strip() for t in args.required.split(',')]
        for token in tokens:
            if not token: continue
            rid = resolve_card_identifier(token, card_data)
            if not rid:
                print(f"ERROR: Could not resolve required card '{token}'")
                return
            base_rid = get_base_card_id(rid)
            print(f"Requiring card: {card_data['id_to_name'].get(base_rid, base_rid)} ({rid})")
            required_card_ids.append(rid)

    excluded_card_ids = set()
    if args.exclude:
        tokens = [t.strip() for t in args.exclude.split(',')]
        for token in tokens:
            if not token: continue
            xid = resolve_card_identifier(token, card_data)
            if not xid:
                print(f"ERROR: Could not resolve excluded card '{token}'")
                return
            base_xid = get_base_card_id(xid)
            print(f"Excluding card: {card_data['id_to_name'].get(base_xid, base_xid)} ({xid})")
            excluded_card_ids.add(xid)
            
    # Check for conflicts
    for rid in required_card_ids:
        base_rid = get_base_card_id(rid)
        if rid in excluded_card_ids or base_rid in excluded_card_ids:
            print(f"ERROR: Card {rid} is both required and excluded.")
            return
    
    print("Loading ML model...")
    if not os.path.exists(MODEL_FILE):
        print(f"ERROR: Model file {MODEL_FILE} not found. Run trainer.py first.")
        return
    
    model = xgb.XGBClassifier()
    model.load_model(MODEL_FILE)
    
    print("Loading feature map...")
    with open(FEATURE_MAP_FILE, 'r') as f:
        feature_map = json.load(f)

    # Load metadata for normalization
    training_max_trophies = 10000 # default
    if os.path.exists(META_FILE):
        with open(META_FILE, 'r') as f:
            metadata = json.load(f)
            training_max_trophies = metadata.get('max_trophies', 10000)
    print(f"Using Training Max Trophies: {training_max_trophies}")
    
    n_features = len(feature_map)
    
    # Fetch player data
    player_tag = args.tag
    print(f"\nFetching player data for {player_tag}...")
    player_data = fetch_player_data(player_tag)
    
    if not player_data:
        print("ERROR: Could not fetch player data")
        return
    
    player_trophies = player_data.get('trophies', 5000)
    print(f"Player Trophies: {player_trophies}")
    
    # Module 1: Build gauntlet
    print("\n" + "=" * 60)
    print("MODULE 1: Building Gauntlet")
    print("=" * 60)
    gauntlet = build_gauntlet(player_trophies, CSV_FILE, trophy_range=500, gauntlet_size=args.gauntlet_size)
    
    # Module 2: Extract inventory
    print("\n" + "=" * 60)
    print("MODULE 2: Player Inventory")
    print("=" * 60)
    inventory, evo_cards, hero_cards = extract_player_inventory(player_data)
    if args.assume_all_evos:
        # Unlock every evolution-capable card
        assumed = set(card_categories['evolutions'])
        evo_cards = evo_cards | assumed
        print(f"Assuming all evos unlocked ({len(evo_cards)} evo-capable cards)")
    if args.assume_all_max_level:
        inventory = {card_id: 16 for card_id in inventory}
        print(f"Assuming all cards are max level (16)")
    print(f"Player has {len(inventory)} cards available")
    print(f"Player has {len(evo_cards)} cards with EVO unlocked")
    print(f"Player has {len(hero_cards)} cards with HERO unlocked")

    for rid in required_card_ids:
        base_rid = get_base_card_id(rid)
        if base_rid not in inventory:
            print(f"ERROR: Required card {card_data['id_to_name'].get(base_rid, base_rid)} is not in your inventory")
            return
        if rid.endswith('_EVO') and base_rid not in evo_cards:
            print(f"ERROR: Required EVO for {card_data['id_to_name'].get(base_rid, base_rid)} is not unlocked")
            return
        if rid.endswith('_HERO') and base_rid not in hero_cards and base_rid not in card_categories['champions']:
            print(f"ERROR: Required HERO for {card_data['id_to_name'].get(base_rid, base_rid)} is not unlocked")
            return
    
    # Evaluate player's most recent trophy road (1v1 Ladder) deck
    print("\n" + "=" * 60)
    print("RECENT TROPHY ROAD DECK")
    print("=" * 60)

    recent_battle = fetch_recent_trophy_road_battle(player_tag)

    if not recent_battle:
        print("No recent trophy road 1v1 battles found.")
    else:
        deck_info = extract_deck_from_battle(recent_battle)

        if not deck_info:
            print("Could not extract a valid 8-card deck from the latest trophy road battle.")
        else:
            deck_ids, deck_levels, evo_upgrades, hero_upgrades, trophy_snapshot, tower_troop_id = deck_info

            if args.assume_all_max_level:
                deck_levels = {card_id: 16 for card_id in deck_levels}

            eval_trophies = trophy_snapshot or player_trophies

            # Convert to deck variant tokens for evaluation
            recent_deck_variants = []
            for cid in deck_ids:
                if cid in evo_upgrades:
                    recent_deck_variants.append(f"{cid}_EVO")
                elif cid in hero_upgrades:
                    recent_deck_variants.append(f"{cid}_HERO")
                else:
                    recent_deck_variants.append(cid)

            # Calculate predicted win rate using the levels from the battle snapshot
            recent_win_rate = calculate_deck_score(
                (recent_deck_variants, tower_troop_id),
                deck_levels,
                gauntlet,
                model,
                feature_map,
                n_features,
                eval_trophies,
                training_max_trophies,
                evo_upgrades,
                hero_upgrades
            )

            id_to_name = card_data.get('id_to_name', {})
            card_categories = identify_card_types(card_data)
            output_lines = []

            for card_id in deck_ids:
                name = id_to_name.get(card_id, f"Unknown({card_id})")
                level = deck_levels.get(card_id, inventory.get(card_id, 0))

                tags = []
                if card_id in hero_upgrades:
                    tags.append("HERO")
                elif card_id in evo_upgrades:
                    tags.append("EVO")
                elif card_id in card_categories['champions']:
                    tags.append("CHAMPION")

                tag_str = f" [{', '.join(tags)}]" if tags else ""
                output_lines.append(f"  - {name} (Level {level}){tag_str}")

            print("\n".join(output_lines))
            
            if tower_troop_id:
                tower_name = id_to_name.get(tower_troop_id, f"Unknown({tower_troop_id})")
                tower_level = deck_levels.get(tower_troop_id, inventory.get(tower_troop_id, 0))
                print(f"  - {tower_name} (Level {tower_level}) [TOWER TROOP]")

            # Calculate elixir average and consistency for recent deck
            avg_elixir = calculate_deck_elixir(deck_ids, card_data)
            consistency = calculate_consistency_score(deck_ids, card_data)
            
            # Validate using player's full inventory and evo/hero capability sets
            if not is_valid_deck((recent_deck_variants, tower_troop_id), inventory, card_categories, evo_cards, hero_cards):
                print("\nNOTE: Deck does not satisfy optimizer constraints (tower troop/evo/hero limits). Showing raw prediction anyway.")

            print(f"\nAverage Elixir: {avg_elixir:.2f}")
            print(f"Consistency Score: {consistency:.1f}/100")
            print(f"Trophies used for evaluation: {eval_trophies}")
            print(f"Predicted Win Rate: {recent_win_rate * 100:.2f}%")
    
    # Module 3 & 4: Optimize
    print("\n" + "=" * 60)
    print("MODULE 3 & 4: Running Genetic Algorithm")
    print("=" * 60)
    
    best_deck, win_rate = genetic_algorithm(
        inventory, card_categories, gauntlet, 
        model, feature_map, n_features, player_trophies, training_max_trophies, evo_cards, hero_cards, required_card_ids, excluded_card_ids, card_data, min_elixir, max_elixir
    )
    
    if best_deck:
        deck_ids = best_deck[0] if isinstance(best_deck, tuple) else best_deck
        avg_elixir = calculate_deck_elixir(deck_ids, card_data)
        consistency = calculate_consistency_score(deck_ids, card_data)
        
        # Calculate UVs (Upgrade Values)
        evaluator = DeckEvaluator(gauntlet, model, feature_map, n_features, player_trophies, training_max_trophies, evo_cards, hero_cards)
        # Re-calculate base score to ensure consistency
        base_scores = evaluator.evaluate([best_deck], inventory)
        base_win_rate = base_scores[0]
        uv_data = calculate_upgrade_values(best_deck, evaluator, inventory, base_win_rate)
        
        print("\n" + "=" * 60)
        print("OPTIMIZED DECK")
        print("=" * 60)
        print(format_deck_output(best_deck, inventory, card_data, evo_cards, hero_cards, upgrade_values=uv_data))
        print(f"\nAverage Elixir: {avg_elixir:.2f}")
        print(f"Consistency Score: {consistency:.1f}/100")
        print(f"Predicted Win Rate: {win_rate * 100:.2f}%")
    else:
        print("\nERROR: Failed to generate valid deck")

if __name__ == "__main__":
    main()
