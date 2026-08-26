# 🐕 24/7 AI Dog System - Quick Start Guide

A complete robotic toy dog system that runs autonomously 24/7 with realistic behaviors, AI decision-making, and natural language control.

## Installation

### Prerequisites
- Python 3.8+
- Ollama with `granite4.1:3b` model (optional, for LLM features)

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# For animation export features (optional)
pip install -r requirements_photos.txt
```

## Quick Start

### 1. Run Demo Mode (No User Input Needed)
```bash
python3 dog_interactive_chat.py --demo
```

Shows 5 predefined commands executing automatically:
- Greeting the dog
- Querying dog status
- Giving action commands
- Checking energy levels

**Output:**
```
🐕 24/7 AI DOG - DEMO SESSION
You: Hello dog!
🐕 Dog: 🐕 The dog perks up excitedly and tail wags!

You: What is the dog doing?
🐕 Dog: 
🐕 DOG STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Current Behavior: ALERT
Energy:     [⚡⚡⚡⚡⚡░░░░░] 50%
Hunger:     [🍗🍗🍗🍗🍗░░░░░] 50%
Attention:  [👀👀👀░░░░░░░] 30%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 2. Interactive Chat Mode
```bash
python3 dog_interactive_chat.py
```

Starts an interactive session where you can chat with the dog:

```
🐕 24/7 AI DOG - INTERACTIVE CHAT
Type 'help' for commands, 'quit' to exit.

You: hello
🐕 Dog: 🐕 Tail wagging enthusiastically! The dog is happy to see you.

You: make the dog play
🐕 Dog: 🎾 The dog gets excited and is ready to play!

You: is the dog tired?
🐕 Dog: 😊 The dog feels good. (50% energy)

You: dog status
🐕 Dog: [full status with metrics]

You: quit
👋 Goodbye! The dog waves its tail as you leave...
```

### 3. Run Tests
```bash
# All Phase tests
python3 test_phase1_foundation.py
python3 test_phase2_orchestrator.py
python3 test_phase3_llm.py
python3 test_phase4_ui.py

# Run all at once
for f in test_phase*.py; do python3 "$f" || break; done
```

## Command Guide

### 🎮 Action Commands
Make the dog do something:
```
"make the dog play"       → Dog plays
"let the dog sleep"       → Dog sleeps
"feed the dog"            → Dog eats
"dog explore"             → Dog explores
"scratch" / "groom"       → Self-care behaviors
"dog should rest"         → Dog rests
"make the dog alert"      → Dog becomes alert
```

### ❓ Query Commands
Ask about the dog:
```
"is the dog tired?"       → Check energy level
"is the dog hungry?"      → Check hunger level
"how's the dog feeling?"  → Check attention/mood
```

### 📊 Status Commands
Get full status:
```
"dog status"              → Show full status report
"what is the dog doing?"  → Current behavior
"how is the dog?"         → Overall check
```

### ℹ️ Help & Navigation
```
"help"                    → Show available commands
"stop" / "quit" / "exit"  → End session
"bye"                     → Say goodbye
```

### 👋 Greetings
The dog recognizes friendly greetings:
```
"hello"     "hi"     "hey"     "greetings"     "howdy"
```

## System Architecture

```
┌─────────────────────────────────────────────────┐
│         PHASE 4: User Interface                 │
│   - Natural language command parser             │
│   - Interactive chat UI                         │
│   - Conversation management                     │
└────────────┬────────────────────────────────────┘
             ↓
┌─────────────────────────────────────────────────┐
│    PHASE 3: LLM-Based Behavior Advisor         │
│   - Intelligent behavior decisions              │
│   - Command interpretation                      │
│   - Contextual responses (with Ollama)          │
└────────────┬────────────────────────────────────┘
             ↓
┌─────────────────────────────────────────────────┐
│   PHASE 2: Behavior Orchestrator                │
│   - Map behaviors to movement sequences         │
│   - Compose complex movements                   │
│   - Handle state transitions                    │
└────────────┬────────────────────────────────────┘
             ↓
┌─────────────────────────────────────────────────┐
│    PHASE 1: Daily Schedule & Runtime            │
│   - 24-hour activity schedule                   │
│   - Metrics tracking (energy/hunger/attention)  │
│   - Async event loop coordination               │
│   - Hardware abstraction layer                  │
└─────────────────────────────────────────────────┘
```

## Core Components

| File | Purpose |
|------|---------|
| `dog_scheduler.py` | Daily schedule, metrics tracking, state management |
| `dog_behavior_orchestrator.py` | Map behaviors to movements |
| `mellea_dog_behavior.py` | LLM-powered decisions (optional) |
| `dog_runtime.py` | Main async coordination loop |
| `dog_command_parser.py` | Parse natural language commands |
| `dog_interactive_chat.py` | Chat interface |
| `hardware_interface.py` | Hardware abstraction (sim/real servo) |
| `dogmove.py` | 8 core movement primitives |

## Dog States

The dog operates in 8 distinct behavioral states:

| State | Description | Typical Activity |
|-------|-------------|------------------|
| 🛌 SLEEPING | Resting/dreaming | Occasional stretches, ear twitches |
| 👀 ALERT | Watchful/attentive | Head tilts, ear flares, looking around |
| 🎾 PLAYING | Active/engaged | Tail wags, walks, paws up, spinning |
| 😌 RESTING | Calm/relaxed | Slow breathing, repositioning |
| 🔍 EXPLORING | Curious/investigating | Varied walks, head turns, sniffing |
| 🍖 EATING | Feeding | Head down, jaw movements, neck positioning |
| 🤔 SCRATCHING | Self-grooming | Contortions, leg movements, head turns |
| 🧼 GROOMING | Hygiene routine | Specific licking/cleaning sequences |

## Metrics System

The dog tracks three continuous metrics (0-100%):

- **⚡ Energy Level**: Decreases with activity, recovers with rest/sleep
- **🍗 Hunger Level**: Increases over time, decreases when eating
- **👀 Attention Level**: How much the dog wants user interaction

These drive behavior decisions and state transitions.

## 24-Hour Schedule

Default daily cycle:

| Time | Behavior | Why |
|------|----------|-----|
| 00:00-08:00 | SLEEPING | Nighttime |
| 08:00-10:00 | ALERT + GROOMING | Morning routine |
| 10:00-13:00 | PLAY + EXPLORE | Morning energy peak |
| 13:00-15:00 | RESTING | Post-lunch rest |
| 15:00-18:00 | PLAY + EXPLORE | Afternoon activity |
| 18:00-20:00 | EATING + REST | Evening |
| 20:00-23:00 | WIND-DOWN | Settling for night |
| 23:00-00:00 | SLEEPING | Sleep prep |

## Python API Usage

### Simple Interactive Session
```python
import asyncio
from dog_interactive_chat import run_interactive_chat

async def main():
    ui = await run_interactive_chat()
    # UI runs in terminal

asyncio.run(main())
```

### Programmatic Command Processing
```python
from dog_command_parser import InteractiveSession
from dog_scheduler import DogState, DogStateMetrics

session = InteractiveSession()
metrics = DogStateMetrics(energy_level=70, hunger_level=40)

# Process a command
cmd_type, target, response = session.process_command(
    "make the dog play",
    DogState.ALERT,
    metrics
)

print(f"Type: {cmd_type}")      # "action"
print(f"Target: {target}")      # DogState.PLAYING
print(f"Response: {response}")  # "🎾 The dog gets excited..."
```

### Access Conversation History
```python
# Get all exchanges
history = session.get_history()
for exchange in history:
    print(f"User: {exchange['user']}")
    print(f"Dog:  {exchange['dog']}")
    print(f"Type: {exchange['command_type']}")

# Get summary
summary = session.get_conversation_summary()
print(f"Total exchanges: {summary['total_exchanges']}")
print(f"Command breakdown: {summary['command_types']}")
```

## Testing

### Run All Tests
```bash
python3 test_phase1_foundation.py  # Scheduler & metrics
python3 test_phase2_orchestrator.py # Behaviors & movements
python3 test_phase3_llm.py         # LLM integration
python3 test_phase4_ui.py          # Chat interface
```

### Expected Output
```
==================================================
✓ ALL TESTS PASSED
==================================================
```

All 26 tests across 4 phases should pass.

## Configuration

### Enable/Disable LLM
Edit the main in `dog_interactive_chat.py`:

```python
# With LLM (requires Ollama running)
ui = await run_interactive_chat(use_llm=True)

# Without LLM (scheduler only)
ui = await run_interactive_chat(use_llm=False)
```

### Adjust Daily Schedule
Edit `dog_scheduler.py`:

```python
HOURLY_BASE_ACTIVITY = {
    0: DogState.SLEEPING,
    6: DogState.ALERT,          # Wake up at 6am
    9: DogState.PLAYING,        # Play time
    13: DogState.RESTING,       # Afternoon nap
    18: DogState.EXPLORING,     # Evening walk
    # ... add more hours
}
```

### Hardware Control
Edit `dog_runtime.py` to use real servo hardware:

```python
from hardware_interface import ServoControlInterface

# Instead of:
hw = SimulationHardwareInterface()

# Use:
hw = ServoControlInterface(port="/dev/ttyUSB0")  # Arduino/Raspberry Pi
```

## Troubleshooting

### Issue: "Module not found" errors
```bash
pip install -r requirements.txt
```

### Issue: LLM not responding
Ensure Ollama is running:
```bash
ollama serve
# In another terminal
ollama pull granite4.1:3b
```

Or run without LLM:
```python
ui = await run_interactive_chat(use_llm=False)
```

### Issue: Tests fail
Check Python version (3.8+) and dependencies installed.

## Next Steps

1. **Try Demo Mode**: `python3 dog_interactive_chat.py --demo`
2. **Run Interactive Chat**: `python3 dog_interactive_chat.py`
3. **Run Tests**: `python3 test_phase4_ui.py`
4. **Explore Code**: Start with `dog_scheduler.py` (Phase 1 foundation)
5. **Customize Behaviors**: Edit `dog_behavior_orchestrator.py`
6. **Add Hardware**: Implement `ServoControlInterface` in `hardware_interface.py`

## Key Files Reference

- **Quick start this file**: `DOG_24_7_QUICKSTART.md`
- **Full architecture**: `ARCHITECTURE.txt`
- **Phase summaries**: `PHASE1_COMPLETE.md`, `PHASE2_COMPLETE.md`, etc.
- **Dog movements**: `README_DOG_MOVEMENTS.md`
- **Activity catalog**: `dog_activities.json`

## System Requirements

- **CPU**: Any modern processor (tested on Apple Silicon M1+)
- **Memory**: 512MB minimum (2GB+ with LLM)
- **Python**: 3.8 or higher
- **Time Accuracy**: System clock should be reasonably accurate for schedule
- **Optional - Ollama**: For LLM-based decision making

## What's Included

✓ Full 24/7 autonomous scheduler  
✓ 8 behavioral states with realistic transitions  
✓ Natural language command parser  
✓ LLM integration (Ollama/Granite)  
✓ Movement orchestration system  
✓ Hardware abstraction (simulation + real servo)  
✓ Conversation management  
✓ Comprehensive test suite (26 tests, all passing)  
✓ Interactive chat interface  
✓ Demo mode for quick testing  

## Version

**Status**: Phase 4 Complete ✓

All 4 implementation phases finished:
- Phase 1: Foundation (scheduler, metrics, runtime)
- Phase 2: Behavior orchestration (states → movements)
- Phase 3: LLM intelligence (decisions, interpretation)
- Phase 4: User interface (chat, commands, responses)

---

**Ready to start?** Run: `python3 dog_interactive_chat.py --demo`

🐕 Enjoy your AI dog!
