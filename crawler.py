import os
import json
import sqlite3
import asyncio
import aiohttp
import requests
import urllib.parse
import argparse
from datetime import datetime, date, timedelta
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

API_TOKEN = os.getenv("TOKEN")
BASE_URL = "https://proxy.royaleapi.dev/v1" # or "https://api.clashroyale.com/v1" if you have a direct token

if not API_TOKEN:
    print("Error: TOKEN not found in .env")
    exit(1)

HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Accept": "application/json"
}

# Config
CONCURRENT_REQUESTS = 10  # Max concurrent requests
RATE_LIMIT_DELAY = 0.1    # Delay between requests

# Database setup
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
DB_FILE = os.path.join(DATA_DIR, "raw_battles.db")

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS battles (
            id TEXT PRIMARY KEY,
            battle_time TEXT,
            game_mode TEXT,
            outcome INTEGER,
            trophy_diff INTEGER,
            player_trophies INTEGER,
            opponent_trophies INTEGER,
            player_loadout TEXT,
            opponent_loadout TEXT,
            crawled_at TEXT
        )
    ''')
    conn.commit()
    return conn

def get_card_id_string(card):
    base_id = str(card['id'])
    evo_level = card.get('evolutionLevel', 0) # either evo or hero
    
    if evo_level == 1:
        return f"{base_id}_EVO"
    elif evo_level == 2:
        return f"{base_id}_HERO"
    return base_id

def parse_loadout(cards, support_cards=None):
    loadout = {}
    all_cards = cards[:]
    if support_cards:
        all_cards.extend(support_cards)
    
    if not all_cards:
        raise ValueError("Loadout is empty")
        
    for card in all_cards:
        if 'id' not in card:
            raise ValueError("Card ID missing")
            
        card_id = get_card_id_string(card)
        
        if 'maxLevel' not in card or 'level' not in card:
            raise ValueError(f"Missing level info for card {card_id}")
            
        max_level = card['maxLevel']
        level = card['level']
        
        normalized_level = level + (16 - max_level) # Hardcoded for level 16 at the moment.
        loadout[card_id] = normalized_level
        
    return json.dumps(loadout)

def first_monday_of_month(year, month):
    first = date(year, month, 1)
    return first + timedelta(days=(0 - first.weekday()) % 7)


def default_season_cutoff(today=None):
    today = today or date.today()
    this_month = first_monday_of_month(today.year, today.month)
    if today >= this_month:
        return this_month
    if today.month == 1:
        return first_monday_of_month(today.year - 1, 12)
    return first_monday_of_month(today.year, today.month - 1)


def parse_cutoff_date(value):
    value = str(value).strip()
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"Invalid date '{value}'. Use DMY format, e.g. 03/08/2026"
    )


def parse_api_battle_date(battle_time):
    try:
        return datetime.strptime(battle_time[:8], "%Y%m%d").date()
    except (TypeError, ValueError, IndexError):
        return None


def extract_battle_data(battle, cutoff_date=None):
    """Parses a single battle into a list of tuples for DB insertion"""
    results = []
    
    # Strict extraction helper
    def get_val(data, key):
        if key not in data:
            raise ValueError(f"Missing key: {key}")
        return data[key]

    try:
        if battle.get('type') != 'PvP': return []
        
        if 'gameMode' not in battle or 'name' not in battle['gameMode']:
            return []
        if battle['gameMode']['name'] != 'Ladder':
            return []

        battle_time = get_val(battle, 'battleTime')
        if cutoff_date is not None:
            battle_date = parse_api_battle_date(battle_time)
            if battle_date is None or battle_date < cutoff_date:
                return []
        
        teams = get_val(battle, 'team')
        opponents = get_val(battle, 'opponent')
        
        if not teams or not opponents:
            return []
        
        player = teams[0]
        opponent = opponents[0]
        
        player_tag = get_val(player, 'tag')
        opponent_tag = get_val(opponent, 'tag')

        p_crowns = get_val(player, 'crowns')
        o_crowns = get_val(opponent, 'crowns')
        
        def get_trophies(data, label):
            if 'startingTrophies' in data:
                return data['startingTrophies']
            elif 'trophyCount' in data:
                return data['trophyCount']
            else:
                raise ValueError(f"Missing trophies for {label}")

        p_trophies = get_trophies(player, "player")
        o_trophies = get_trophies(opponent, "opponent")
        
        p_cards = get_val(player, 'cards')
        o_cards = get_val(opponent, 'cards')
        
        p_support = get_val(player, 'supportCards')
        o_support = get_val(opponent, 'supportCards')
        
        p_loadout_str = parse_loadout(p_cards, p_support)
        o_loadout_str = parse_loadout(o_cards, o_support)

        crawled_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Determine outcome
        if p_crowns > o_crowns:
            p_outcome = 1
            o_outcome = 0
        elif p_crowns < o_crowns:
            p_outcome = 0
            o_outcome = 1
        else:
            return [] # Skip ties

        # Record 1: Player perspective
        row_id_1 = f"{battle_time}_{player_tag}"
        record_1 = (
            row_id_1, battle_time, "PvP", p_outcome, p_trophies - o_trophies,
            p_trophies, o_trophies, p_loadout_str, o_loadout_str, crawled_at
        )
        results.append(record_1)

        # Record 2: Opponent perspective (mirroring)
        # You probably dont explicitly need mirrored data, but I had issues beforehand from non-symetric data.
        row_id_2 = f"{battle_time}_{opponent_tag}"
        record_2 = (
            row_id_2, battle_time, "PvP", o_outcome, o_trophies - p_trophies,
            o_trophies, p_trophies, o_loadout_str, p_loadout_str, crawled_at
        )
        results.append(record_2)
        
        return results

    except ValueError:
        return []
    except Exception as e:
        print(f"Error parsing battle: {e}")
        return []

def parse_limit(value):
    if not value:
        return None
    # Allow whitespace and uppercase suffixes (e.g. '750K', ' 1m')
    value = str(value).strip().lower()
    if value.endswith('k'):
        return int(float(value[:-1]) * 1000)
    elif value.endswith('m'):
        return int(float(value[:-1]) * 1000000)
    return int(value)

class AsyncCrawler:
    def __init__(self, start_clans, limit=None, cutoff_date=None):
        self.limit = limit
        self.cutoff_date = cutoff_date
        self.total_saved = 0
        self.running = True
        self.clan_queue = asyncio.Queue()
        for tag in start_clans:
            t = tag.strip()
            if t and not t.startswith('#'):
                t = '#' + t
            self.clan_queue.put_nowait(t)
            
        self.player_queue = asyncio.Queue()
        self.crawled_players = set()
        self.crawled_clans = set()
        self.battle_buffer = []
        self.buffer_lock = asyncio.Lock()
        
        self.session = None
        self.db_conn = None
        
    async def start(self):
        self.db_conn = init_db()
        
        connector = aiohttp.TCPConnector(limit=CONCURRENT_REQUESTS)
        async with aiohttp.ClientSession(connector=connector, headers=HEADERS) as session:
            self.session = session
            
            # Start workers
            workers = []
            # Clan workers (fewer needed)
            for _ in range(2):
                workers.append(asyncio.create_task(self.clan_worker()))
            # Player workers (main workload)
            for _ in range(8):
                workers.append(asyncio.create_task(self.player_worker()))
            # DB Saver task
            saver = asyncio.create_task(self.db_saver())
            
            await asyncio.gather(*workers)
            await saver

    async def fetch(self, url):
        try:
            async with self.session.get(url) as response:
                if response.status == 429:
                    print("Rate limited (429). Sleeping 10s...")
                    await asyncio.sleep(10)
                    return None
                if response.status != 200:
                    text = await response.text()
                    print(f"Error {response.status} fetching {url}: {text[:200]}")
                    return None
                return await response.json()
        except Exception as e:
            print(f"Exception fetching {url}: {e}")
            return None

    async def clan_worker(self):
        while self.running:
            try:
                clan_tag = await asyncio.wait_for(self.clan_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if clan_tag in self.crawled_clans:
                self.clan_queue.task_done()
                continue
                
            # print(f"Crawling Clan: {clan_tag}")
            url = f"{BASE_URL}/clans/{urllib.parse.quote(clan_tag)}/members"
            data = await self.fetch(url)
            
            if not data:
                print(f"Failed to fetch clan {clan_tag} or no data returned")
            
            if data:
                members = data.get("items", [])
                print(f"Clan {clan_tag}: Found {len(members)} members")
                for m in members:
                    tag = m.get('tag')
                    if tag and tag not in self.crawled_players:
                        await self.player_queue.put(tag)
            
            self.crawled_clans.add(clan_tag)
            self.clan_queue.task_done()
            await asyncio.sleep(RATE_LIMIT_DELAY)

    async def player_worker(self):
        while self.running:
            try:
                player_tag = await asyncio.wait_for(self.player_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            if player_tag in self.crawled_players:
                self.player_queue.task_done()
                continue
                
            # print(f"  Crawling Player: {player_tag}")
            url = f"{BASE_URL}/players/{urllib.parse.quote(player_tag)}/battlelog"
            battles = await self.fetch(url)
            if not battles:
                print(f"Failed to fetch player {player_tag} battlelog or no data returned")

            if battles:
                count = 0
                for battle in battles:
                    records = extract_battle_data(battle, cutoff_date=self.cutoff_date)
                    if records:
                        async with self.buffer_lock:
                            self.battle_buffer.extend(records)
                        count += 1
                        # Discover more clans
                        try:
                            teams = battle.get('team') or []
                            opponents = battle.get('opponent') or []
                            
                            # Add clans
                            if teams:
                                clan = (teams[0].get('clan') or {}).get('tag')
                                if clan and clan not in self.crawled_clans:
                                    if len(self.clan_queue._queue) < 100:
                                        await self.clan_queue.put(clan)
                            
                            if opponents:
                                opp_tag = opponents[0].get('tag')
                                if opp_tag and opp_tag not in self.crawled_players:
                                    # Add opponent to player queue occasionally
                                    if len(self.player_queue._queue) < 1000:
                                        await self.player_queue.put(opp_tag)

                        except Exception:
                            pass
                
                if count > 0:
                    print(f"  Processed {player_tag}: {count} battles")
            
            self.crawled_players.add(player_tag)
            self.player_queue.task_done()
            await asyncio.sleep(RATE_LIMIT_DELAY)

    async def db_saver(self):
        """Periodically flush buffer to DB"""
        while self.running or self.battle_buffer:
            await asyncio.sleep(2)
            async with self.buffer_lock:
                if not self.battle_buffer:
                    if not self.running:
                        break
                    continue
                
                batch = self.battle_buffer[:]
                self.battle_buffer.clear()
            
            if batch:
                try:
                    cursor = self.db_conn.cursor()
                    cursor.executemany('''
                        INSERT OR IGNORE INTO battles (
                            id, battle_time, game_mode, outcome, trophy_diff, 
                            player_trophies, opponent_trophies, player_loadout, 
                            opponent_loadout, crawled_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', batch) # ? as placeholders
                    self.db_conn.commit()
                    self.total_saved += len(batch)
                    print(f">> Saved batch of {len(batch)} records. Total: {self.total_saved}/{self.limit if self.limit else 'Inf'}")
                    
                    if self.limit and self.total_saved >= self.limit:
                        print("Limit reached. Stopping.")
                        self.running = False
                        
                except sqlite3.Error as e:
                    print(f"Database error saving batch: {e}")

def card_type_from_id(card_id):
    """
    Derive card type from Card ID (stable across updates):
    """
    cid = str(card_id)
    if cid.startswith("159"):
        return "tower_troop"
    if cid.startswith("270"):
        return "building"
    if cid.startswith("280"):
        return "spell"
    return "troop" # other


def _ingest_card_item(card_data, item):
    """Write one API card/support item into the card_data dict."""
    card_id = str(item["id"])
    name = item["name"]
    rarity = item.get("rarity", "common")
    elixir = item.get("elixirCost", 0)

    card_data["id_to_name"][card_id] = name
    card_data["name_to_id"][name] = card_id
    card_data["rarities"][name] = rarity
    card_data["elixir_costs"][name] = elixir
    card_data["types"][card_id] = card_type_from_id(card_id)

    max_evo = item.get("maxEvolutionLevel")
    if max_evo is not None:
        card_data["max_evolution_level"][card_id] = int(max_evo)
        if int(max_evo) >= 1:
            card_data["can_evolve"].append(card_id)


def update_card_data():
    print("Fetching card data from API")
    url = f"{BASE_URL}/cards"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            data = response.json()
            items = data.get("items", [])
            support_items = data.get("supportItems", [])

            card_data = {
                "id_to_name": {},
                "name_to_id": {},
                "rarities": {},
                "elixir_costs": {},
                "types": {},                 # card_id - troop|building|spell|tower_troop
                "can_evolve": [],            # card_ids with an evolution path
                "max_evolution_level": {},   # card_id - maxEvolutionLevel from API
            }

            for item in items:
                _ingest_card_item(card_data, item)

            # Tower troops come from supportItems
            for item in support_items:
                _ingest_card_item(card_data, item)

            card_data_file = os.path.join(DATA_DIR, "cardData.json")
            with open(card_data_file, 'w', encoding='utf-8') as f:
                json.dump(card_data, f, indent=4)
            n_buildings = sum(1 for t in card_data["types"].values() if t == "building")
            n_towers = sum(1 for t in card_data["types"].values() if t == "tower_troop")
            n_champs = sum(1 for r in card_data["rarities"].values() if r == "champion")
            print(
                f"Updated cardData.json with {len(items)} cards + "
                f"{len(support_items)} tower troops "
                f"({n_buildings} buildings, {n_towers} tower troops, "
                f"{n_champs} champions, {len(card_data['can_evolve'])} evo-capable)."
            )
        else:
            print(f"Warning: Failed to update card data. Status {response.status_code}")
    except Exception as e:
        print(f"Warning: Exception while updating card data: {e}")

def main():
    parser = argparse.ArgumentParser(description="Clash Royale Battle Crawler")
    parser.add_argument("--limit", type=str, help="Limit number of battles to crawl (e.g., 750k, 1m)")
    parser.add_argument(
        "--cutoff-date",
        type=parse_cutoff_date,
        default=None,
        help="Only collect battles on or after this date (DMY, e.g. 03/08/2026). "
             "Defaults to the first Monday of the current month (season start).",
    )
    args = parser.parse_args()

    limit = parse_limit(args.limit)
    cutoff_date = args.cutoff_date or default_season_cutoff()

    # Initial seed clans are read ONLY from .env (SEED_CLAN_IDS)
    seed_clans_env = os.getenv("SEED_CLAN_IDS")
    if not seed_clans_env:
        print("Error: SEED_CLAN_IDS not set in .env. Please add seed clan tags (comma-separated).")
        exit(1)

    seed_clans = [s.strip() for s in seed_clans_env.split(",") if s.strip()]
    if not seed_clans:
        print("Error: SEED_CLAN_IDS in .env contains no valid entries.")
        exit(1)

    print("Starting Crawler")
    
    # Update central card dictionary before crawling
    update_card_data()
    
    if limit:
        print(f"Battle limit set to: {limit}")
    print(f"Cutoff date: {cutoff_date.strftime('%d/%m/%Y')} (battles on or after this date)")
    print(f"Concurrent Requests: {CONCURRENT_REQUESTS}")
    
    crawler = AsyncCrawler(seed_clans, limit=limit, cutoff_date=cutoff_date)
    
    try:
        asyncio.run(crawler.start())
    except KeyboardInterrupt:
        print("\nCrawler stopped by user.")

if __name__ == "__main__":
    main()
