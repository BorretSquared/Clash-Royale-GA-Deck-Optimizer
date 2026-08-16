# Clash Royale Genetic Algorithm Deck Optimizer

A complete machine learning model with xgboost for crawling Clash Royale battle logs, analyzing data, and training a genetic algorithm for battle outcomes, ultimately being used to recommend optimal decks for use on the trophy road.

## Notes/Good-To-Knows

* An accuracy of ~0.65 and ROC AUC of ~0.71 on estimating the winner of a matchup with around 2 million matches as a baseline.
* The xgboost model and data are not shared due to game meta shifts, card additions, balance changes, and seasonal updates. When changes are made, the model should be retrained and new data should be collected; simply delete the `data` directory and `xgboost_model.json` to restart. For optimal accuracy, the model should be retrained mid-season to account for meta shifts and new cards, using the `--cutoff-date` flag after patches are added.
* **Higher trophy ranges** are recommended because of the way crawling games works in Clash Royale and the lack of a trophy gate implementation currently in `deck.py`. Please describe experiences with deck recommendations in lower trophy ranges on the discussions tab of the repo.
* It's suggested to use this tool after around 5500 trophies as a lower bound. If collecting data for ranges under 5500, around 5 million battles is a reasonable amount due to the lack of available matches. Reference `tests/player_trophy_distribution.png` for where data is lacking most.
* This repo has only been tested on Linux. Please report issues on other OSes.
* Card names should be written with spaces with the whole value in quotes.
* It is **highly recommended** to put season boosted cards in your `.env` for optimal recommendations. Season boost (and therefore king tower) level is assumed from your highest tower troop level.
* Lines 33-38 of `deck.py` can probably be adjusted for different tuning goals.
* This project is designed to function on systems with low RAM and has been tested on both 8GB and 32GB systems.
* `deck.py`'s `UV` value denotes how much the predicted win rate is expected to increase by upgrading that specific card.

## Prerequisites

* Python 3.8+
* A Clash Royale API Token (or read below)

## Setup

1. **Create and activate a virtual environment if needed (Linux)**

   Create a Python environment before installing dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   If `.venv` already exists, just activate it with the second command.

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**

   Remove the `.example` from `.env.example` to make a `.env` file in the root directory.

   Get your API token:

   1. Go to [https://developer.clashroyale.com/](https://developer.clashroyale.com/)
   2. Click `Register` or `Log In` to make/sign in to your account.
   3. Click your user in the top right dropdown > My Account.
   4. Click `Create New Key` in the bottom right.
   5. Select a name and description.
   6. Input the IP address `45.79.218.79` for RoyaleAPI's proxy service: [https://docs.royaleapi.com/proxy.html](https://docs.royaleapi.com/proxy.html)
   7. Click Create Key.
   8. Click on your named key in the right side panel.
   9. Copy and paste your token into your `.env` file. You can also add boosted cards there if you want to account for seasonal boosts.
   10. Add a clan tag to `SEED_CLAN_IDS` for the crawler to use. Either use your own clan tag, find a clan on RoyaleAPI, or find one in-game. Exclude hashtags.

   Alternatively, you can simply change the `BASE_URL` in `crawler.py` and `deck.py` to the official Clash Royale API and whitelist your IP there instead.

## Usage

Steps to go from collecting data to a trained model:

### 1. Crawl Data

Fetch battle logs and store them in a local SQLite database (`data/raw_battles.db`).

```bash
python crawler.py --limit 1m
python crawler.py --limit 1m --cutoff-date 03/08/2026
```

1 million crawls is a recommended starting amount. Going to 2 million usually improves the model somewhat, while much greater amounts generally have diminishing returns.

Only battles on or after the cutoff date are stored. `--cutoff-date` is DMY (`DD/MM/YYYY`). If omitted, the cutoff is the first Monday of the current month (season start); if that Monday has not occurred yet, the previous month's first Monday is used.

Press `Ctrl + C` to stop the crawler when wanted/needed.

### 2. Analyze & Export

Export the crawled data from the database to a CSV file (`data/export_battles.csv`).

```bash
python analyzer.py
```

### 3. Vectorize Data

Process the CSV data into NumPy arrays for training.

```bash
python vectorizer.py
```

### 4. Train Model

Train an XGBoost model on the vectorized data.

```bash
python trainer.py
```

### 5. Use Model

Use the trained model to recommend decks for a specific player tag. Go to [https://royaleapi.com/](https://royaleapi.com/) to easily find yours.

```bash
python deck.py --tag TAGHERE
```

The CLI also supports filters and forcing specific cards to be included/excluded:

```bash
usage: deck.py [-h] [--required REQUIRED] [--exclude EXCLUDE] [--tag TAG] [--elixir ELIXIR] [--assume-all-evos] [--assume-all-max-level]

Clash Royale Deck Optimizer

options:
  -h, --help            show this help message and exit
  --required REQUIRED   Card ID or name to force include (supports 'evo ' / 'hero ' prefixes)
  --exclude EXCLUDE     Card ID or name to force exclude (supports 'evo ' / 'hero ' prefixes)
  --tag TAG             Player tag to analyze (default: CLL0LCPPJ)
  --elixir ELIXIR       Elixir range filter (e.g., '2.5-5.0', '2.5-', or '-5.0')
  --assume-all-evos     Treat every evolution-capable card as unlocked (heroes stay heroes)
  --assume-all-max-level
```

Files you should run in order:

```text
crawler.py
    ↓
analyzer.py
    ↓
vectorizer.py
    ↓
trainer.py
    ↓
deck.py
```

## Disclaimer

This material is unofficial and is not endorsed by Supercell. For more information see Supercell's Fan Content Policy: [www.supercell.com/fan-content-policy](http://www.supercell.com/fan-content-policy).
