# TOMLab Diplomacy

A local Diplomacy-style AI sandbox for testing whether knowledge-graph support improves LLM agent performance in a multi-agent strategic game.

The main experiment is simple:

> Give some players access to a knowledge graph / memory substrate, remove it from other players, and compare performance over play.

Diplomacy is a useful test environment because it is not just a tactical board game. It stresses negotiation, trust, betrayal, alliances, long-horizon planning, memory of prior commitments, and theory-of-mind reasoning.

## Core Question

Does a knowledge graph help an LLM agent play a complex social-strategic game better?

This project is designed to let the user compare agents with and without KG support under similar conditions.

Examples of questions this sandbox can explore:

- Do KG-enabled players make more coherent long-term plans?
- Do they remember prior alliances and betrayals better?
- Do they negotiate more consistently?
- Do they issue better orders?
- Do they survive longer or gain more supply centers?
- Does KG help all powers equally, or only some?
- Does KG help strong models but confuse weak models?
- Does removing KG from selected players create measurable performance differences?

## Current Status

This is a runnable local research app.

The app provides a Flask-based local interface for running and inspecting Diplomacy-style agent sessions. It is intended for hands-on experimentation, KG ablation testing, and agent-behavior inspection.

## Recommended Model Strength

Use a model at least as capable as Claude Haiku-class or better.

Weaker models may run, but they are more likely to:

- produce invalid or poorly formatted moves
- lose track of the board state
- misunderstand Diplomacy orders
- fail to maintain coherent alliances
- make noisy results harder to interpret

For meaningful KG-vs-no-KG comparison, the base model should be strong enough to play the game at a basically competent level without the KG. Otherwise the experiment mostly measures model failure rather than KG contribution.

## What This Repo Tests

This repo is mainly a KG ablation lab.

The key comparison is:

```text
LLM player + KG / memory support
vs.
LLM player without KG / memory support
```

The KG can be selectively enabled or disabled for particular players. This allows experiments such as:

```text
England: KG enabled
France: KG disabled
Germany: KG enabled
Russia: KG disabled
Austria: KG disabled
Italy: KG enabled
Turkey: KG disabled
```

The goal is to observe whether KG-enabled agents show better strategic persistence, negotiation memory, alliance tracking, and board performance.

## Why Diplomacy?

Diplomacy is valuable for agent testing because success depends on more than local tactical search.

A competent Diplomacy agent needs to reason about:

- geography
- timing
- alliance formation
- promises
- deception
- betrayal
- trust repair
- multi-turn planning
- other agents' likely motives
- the gap between what players say and what they do

That makes it a better testbed for memory and theory-of-mind systems than many small board games.

## Features

Depending on configuration, the app supports:

- local Flask web interface
- Diplomacy-style game/session flow
- LLM-driven player behavior
- selective KG enable/disable by player
- prompt inspection
- raw model response inspection
- session state tracking
- move/order generation
- game-state visualization
- experimental agent/session variants

## Requirements

- Python 3.10+
- pip
- Terminal / command line
- Optional: Anthropic API key or other configured LLM provider

## Setup

Clone the repository:

```bash
git clone https://github.com/DormantOne/tomlab-diplomacy.git
cd tomlab-diplomacy
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## API Keys

If using Anthropic-backed agents, set:

```bash
export ANTHROPIC_API_KEY="your_key_here"
```

On Windows PowerShell:

```powershell
$env:ANTHROPIC_API_KEY="your_key_here"
```

Do not put real API keys directly into the source code.

## Run

Start the app:

```bash
python run.py
```

Then open:

```text
http://127.0.0.1:5050
```

If port `5050` is already in use, change the port in the launcher or server configuration.

## Suggested Experiment Design

For cleaner results, avoid judging from a single game.

A better pattern is:

1. Run several games.
2. Rotate which countries have KG enabled.
3. Keep the same base model across conditions.
4. Compare KG-enabled and KG-disabled players.
5. Track supply centers, survival, invalid orders, alliance coherence, and final board position.
6. Repeat with enough trials that one lucky opening does not dominate the result.

A simple comparison table might include:

```text
Game ID
Power / country
Model used
KG enabled: yes/no
Final supply center count
Turns survived
Invalid orders
Notable alliance behavior
Final outcome
```

## Interpreting Results

A KG-enabled player doing well in one game is interesting but not proof.

Stronger evidence would come from repeated games where KG-enabled agents, across different countries and starting positions, show better average performance than KG-disabled agents using the same base model.

Useful signs that KG may be helping:

- fewer repeated strategic mistakes
- better memory of prior agreements
- more consistent alliance behavior
- better adaptation after betrayal
- fewer contradictions between stated plans and moves
- better long-term positioning
- improved survival or supply center count

Useful signs that KG may be hurting:

- overcommitment to stale plans
- confused or irrelevant memory retrieval
- worse tactical orders
- excessive narrative reasoning without board benefit
- worse performance than non-KG agents using the same model

## Approximate Project Structure

```text
.
├── run.py                  # Main launcher
├── run_v2.py               # Alternate launcher / variant
├── requirements.txt        # Python dependencies
├── agents/                 # Agent logic
├── diplomacy_engine/       # Game / adjudication logic
├── server/                 # Flask server and UI routes
│   ├── app.py
│   ├── live.py
│   ├── live_session.py
│   ├── raw_llm_agent.py
│   ├── session.py
│   ├── snapshot.py
│   ├── static/
│   └── templates/
└── README.md
```

## Project Framing

This is not a claim that KG agents are automatically better.

It is a lab for testing that claim.

The important feature is not merely that LLMs can play Diplomacy-style turns. The important feature is the ability to selectively remove or add memory/KG support and observe whether that changes performance.

## License

This project is released under the MIT License.
