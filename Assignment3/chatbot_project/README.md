# CourseBot — ChatterBot Q&A Terminal Client

**MSCS-633-A01 Advanced Artificial Intelligence | Hands-On Assignment 3**  
**Author:** Saru Bhandari
**University:** University of the Cumberlands

---

## Overview

A terminal-based Q&A chatbot built with **Django** and **ChatterBot** (machine-learning conversational dialog engine). The bot trains on the built-in English corpus and responds to natural-language input directly in the console.

### Sample Session

```
user: Good morning! How are you doing?
bot: I am doing very well, thank you for asking.
user: What is artificial intelligence?
bot: Artificial intelligence is the simulation of human intelligence in machines.
user: exit
bot: Goodbye! Have a great day!
```

---

## Project Structure

```
chatbot_project/
├── chatbot/
│   ├── __init__.py
│   └── bot.py              # ChatterBot setup and training logic
├── chatbot_config/
│   ├── __init__.py
│   ├── settings.py         # Django project settings
│   ├── urls.py
│   └── wsgi.py
├── terminal_client.py      # Main runnable terminal chat script
├── requirements.txt        # Manifest / dependency list
└── README.md
```

---

## Requirements

- Python 3.9+
- pip

---

## Setup & Run

```bash
# 1. Clone the repository
git clone https://github.com/SaruBhandari97/MSCS-633-A01_AdvancedAI.git
cd chatbot_project

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Apply Django migrations
python manage.py migrate

# 5. Start the terminal chat client
python terminal_client.py
```

The bot will train on first launch (~30 seconds), then the chat prompt appears.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `Django>=4.2` | Project scaffolding and settings management |
| `ChatterBot==1.2.13` | ML-based conversational dialog engine |
| `chatterbot_corpus==1.2.0` | English training corpus |
| `nltk>=3.8` | Natural language processing (ChatterBot dependency) |
| `SQLAlchemy>=1.4` | Database ORM for response storage |

---

## How It Works

1. **`bot.py`** — Initialises a `ChatBot` instance with `BestMatch` logic adapter and trains it against ChatterBot's English corpus (greetings, conversations, trivia).
2. **`terminal_client.py`** — Bootstraps Django settings, calls `create_bot()`, then enters an interactive `input()` loop. User messages are passed to `bot.get_response()` and the reply is printed.
3. A **confidence threshold of 0.90** ensures the bot uses a polite fallback message rather than a poorly-matched low-confidence answer.

---

## References

- [ChatterBot Documentation](https://chatterbot.readthedocs.io/en/stable/)
- [Django Documentation](https://www.djangoproject.com/)
- [ChatterBot Django Integration](http://chatterbot.readthedocs.io/en/stable/django/index.html)
