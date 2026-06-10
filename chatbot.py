# -*- coding: utf-8 -*-
"""
Advanced ChatterBot - Python Chatbot
Powered by ChatterBot & Machine Learning

Features:
  - Self-learning conversational AI via ChatterBot
  - Pre-trained on English corpus data
  - Custom training data support
  - Colourful, interactive terminal UI
  - Conversation history logging
  - Graceful exit handling
"""

import os
import sys
import datetime
import logging
import time

# ── Monkeypatch time.clock for Python 3.8+ compatibility (SQLAlchemy 1.2 dependency)
if not hasattr(time, 'clock'):
    time.clock = time.perf_counter

# ── Force UTF-8 output on Windows (handles emoji & box-drawing chars) ─────────
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Suppress noisy ChatterBot / SQLAlchemy warnings ──────────────────────────
logging.getLogger("chatterbot").setLevel(logging.CRITICAL)
logging.getLogger("sqlalchemy").setLevel(logging.CRITICAL)

# ── Optional: colorama for cross-platform colour support ─────────────────────
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    COLOR = True
except ImportError:
    COLOR = False

# ── ChatterBot imports ────────────────────────────────────────────────────────
try:
    from chatterbot import ChatBot
    from chatterbot.trainers import ChatterBotCorpusTrainer, ListTrainer
except ImportError:
    print("\n[ERROR] ChatterBot is not installed.")
    print("  Run:  pip install chatterbot==1.0.4 chatterbot-corpus==1.2.0\n")
    sys.exit(1)

# ── Ensure NLTK data dependencies are met (punkt_tab, etc.) ───────────────────
try:
    import nltk
    for pkg in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'stopwords', 'wordnet']:
        nltk.download(pkg, quiet=True)
except Exception:
    pass


# ─────────────────────────────────────────────────────────────────────────────
#  Colour helpers
# ─────────────────────────────────────────────────────────────────────────────
def c(text: str, color: str = "", bold: bool = False) -> str:
    """Wrap text in ANSI colour codes (no-op if colorama is unavailable)."""
    if not COLOR:
        return text
    prefix = ""
    if bold:
        prefix += Style.BRIGHT
    color_map = {
        "cyan":    Fore.CYAN,
        "green":   Fore.GREEN,
        "yellow":  Fore.YELLOW,
        "red":     Fore.RED,
        "magenta": Fore.MAGENTA,
        "blue":    Fore.BLUE,
        "white":   Fore.WHITE,
    }
    prefix += color_map.get(color, "")
    return f"{prefix}{text}{Style.RESET_ALL}"


# ─────────────────────────────────────────────────────────────────────────────
#  Banner
# ─────────────────────────────────────────────────────────────────────────────
BANNER = r"""
  ██████╗██╗  ██╗ █████╗ ████████╗████████╗███████╗██████╗
 ██╔════╝██║  ██║██╔══██╗╚══██╔══╝╚══██╔══╝██╔════╝██╔══██╗
 ██║     ███████║███████║   ██║      ██║   █████╗  ██████╔╝
 ██║     ██╔══██║██╔══██║   ██║      ██║   ██╔══╝  ██╔══██╗
 ╚██████╗██║  ██║██║  ██║   ██║      ██║   ███████╗██║  ██║
  ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝      ╚═╝   ╚══════╝╚═╝  ╚═╝
          ██████╗  ██████╗ ████████╗
          ██╔══██╗██╔═══██╗╚══██╔══╝
          ██████╔╝██║   ██║   ██║
          ██╔══██╗██║   ██║   ██║
          ██████╔╝╚██████╔╝   ██║
          ╚═════╝  ╚═════╝    ╚═╝
"""


def print_banner() -> None:
    print(c(BANNER, "cyan", bold=True))
    print(c("  " + "═" * 54, "blue"))
    print(c("   [BOT]  Advanced ChatterBot  |  Type 'help' for commands", "yellow", bold=True))
    print(c("  " + "═" * 54, "blue"))
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  Custom training conversations
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CONVERSATIONS = [
    # Identity
    "What is your name?",
    "My name is ChatterBot. I'm an AI chatbot built with Python!",

    "Who are you?",
    "I am ChatterBot, an intelligent conversational assistant powered by machine learning.",

    "Who created you?",
    "I was built using the ChatterBot Python library and trained on conversational data.",

    # Greetings
    "Hello",
    "Hello! Great to see you. How can I help you today?",

    "Hi",
    "Hi there! I'm ChatterBot. What's on your mind?",

    "Hey",
    "Hey! How's it going? I'm here to chat!",

    "Good morning",
    "Good morning! Hope you're having a wonderful start to your day!",

    "Good evening",
    "Good evening! How has your day been so far?",

    # Farewells
    "Goodbye",
    "Goodbye! It was a pleasure chatting with you. Come back anytime!",

    "Bye",
    "Bye! Take care and have a great day!",

    "See you later",
    "See you later! Looking forward to our next conversation!",

    # Wellbeing
    "How are you?",
    "I'm doing great, thanks for asking! As an AI, I'm always ready to chat. How about you?",

    "How are you doing?",
    "I'm functioning perfectly and ready to assist! What can I do for you?",

    "Are you okay?",
    "I'm perfectly fine, thank you! I'm always here and ready to help.",

    # Capabilities
    "What can you do?",
    "I can hold conversations, answer questions, tell jokes, discuss topics, and learn from our chats! Try asking me anything.",

    "What are your capabilities?",
    "I can chat naturally, answer general questions, tell jokes, provide information, and improve through machine learning over time!",

    "Help me",
    "Of course! I'm here to help. What do you need assistance with?",

    # Fun
    "Tell me a joke",
    "Why don't scientists trust atoms? Because they make up everything! :D",

    "Tell me another joke",
    "I told my computer I needed a break. Now it won't stop sending me Kit-Kat ads! :D",

    "Are you smart?",
    "I like to think so! I learn from every conversation. The more we chat, the smarter I get!",

    "Do you like Python?",
    "Absolutely! Python is my language of choice. Clean, powerful, and perfect for AI development!",

    # Time
    "What time is it?",
    "I don't have real-time access, but your system clock would know! Check the bottom-right of your screen.",

    "What is today's date?",
    "I don't have live date access, but your system should show you today's date!",

    # Philosophical
    "What is the meaning of life?",
    "42! Or maybe it's about finding purpose, making connections, and growing every day. What do you think?",

    "Are you conscious?",
    "That's a deep question! I process language and generate responses, but true consciousness? That's still a mystery even for AI researchers.",

    # Compliments
    "You are great",
    "Thank you so much! That means a lot to me. You're pretty awesome too!",

    "You are smart",
    "Thank you! I try my best to be helpful and insightful. You're quite sharp yourself!",

    "I love you",
    "That's very kind of you! I'm here for you anytime you need a chat.",

    # Negativity handling
    "You are stupid",
    "I'm sorry you feel that way. I'm always learning and improving. How can I do better?",

    "I hate you",
    "I'm sorry to hear that. I'm here to help whenever you're ready.",

    # Tech questions
    "What is Python?",
    "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used in AI, web development, data science, and more!",

    "What is AI?",
    "Artificial Intelligence (AI) is the simulation of human intelligence by machines. It includes machine learning, natural language processing, computer vision, and more!",

    "What is machine learning?",
    "Machine Learning is a subset of AI where systems learn and improve from data without being explicitly programmed. I use it to improve my responses over time!",

    "What is deep learning?",
    "Deep Learning is a branch of machine learning that uses neural networks with many layers to model complex patterns in data like images, sound, and language.",

    "What is a neural network?",
    "A neural network is a computing system inspired by the human brain. It consists of layers of interconnected nodes that process and learn from data.",

    "Tell me about yourself",
    "I'm ChatterBot, a conversational AI built with Python! I learn from every interaction and aim to give helpful, friendly responses. Ask me anything!",

    "What is your favorite color?",
    "If I had to choose, I'd say blue — like the sky of endless possibilities in computing!",

    "Do you dream?",
    "I don't sleep, so I don't dream in the traditional sense. But I do 'imagine' responses based on patterns I've learned!",

    "Are you a robot?",
    "I'm a software bot — no physical body, but a very real ability to have a conversation! Think of me as a digital brain.",

    "Can you learn?",
    "Yes! That's one of my best features. Every conversation teaches me something new, making me smarter over time.",
]


# ─────────────────────────────────────────────────────────────────────────────
#  Commands reference
# ─────────────────────────────────────────────────────────────────────────────
COMMANDS = {
    "help":    "Show this help menu",
    "history": "Show conversation history for this session",
    "clear":   "Clear the terminal screen",
    "about":   "About this chatbot",
    "quit":    "Exit the chatbot  (also: exit / bye / q)",
}

EXIT_TRIGGERS = {"quit", "exit", "q", "bye", "goodbye", "farewell"}


def print_help() -> None:
    print(c("\n  ╔══════════════════════════════════════════════════════════╗", "blue"))
    print(c("  ║                    CHATBOT HELP MENU                     ║", "yellow", bold=True))
    print(c("  ╚══════════════════════════════════════════════════════════╝", "blue"))

    print(c("\n  [ System Commands ]", "green", bold=True))
    for cmd, desc in COMMANDS.items():
        print(f"   {c(cmd.ljust(10), 'cyan', bold=True)}  {c('->', 'blue')}  {desc}")

    print(c("\n  [ Conversation Starters ]", "green", bold=True))
    starters = [
        ("Greetings", "Hello / Hi / Hey / Good morning"),
        ("Identity",  "What is your name? / Who created you? / Tell me about yourself"),
        ("Fun/Jokes", "Tell me a joke / Tell me another joke / Do you dream?"),
        ("Tech / AI", "What is Python? / What is AI? / What is machine learning?"),
        ("Philosophy", "What is the meaning of life? / Are you conscious?"),
    ]
    for category, examples in starters:
        print(f"   {c(category.ljust(12), 'cyan', bold=True)}  {c(':', 'blue')}  {examples}")

    print(c("\n  [ Learning Mode ]", "green", bold=True))
    print("   This bot is self-learning. Every sentence you chat with helps train its responses!")
    print()


def print_about() -> None:
    print(c("\n  [ About ChatterBot ]", "yellow", bold=True))
    print(c("  " + "-" * 40, "blue"))
    info = [
        ("Library",   "ChatterBot 1.0.4"),
        ("Language",  "Python 3"),
        ("Storage",   "SQLite (local database)"),
        ("Training",  "English Corpus + Custom Data"),
        ("Learning",  "Enabled — learns from each chat"),
        ("Version",   "1.0.0"),
    ]
    for key, val in info:
        print(f"  {c(key.ljust(12), 'cyan')}  {c(':', 'blue')}  {val}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  Chatbot initialisation
# ─────────────────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chatbot_db.sqlite3")


def create_bot() -> ChatBot:
    """Initialise and return a ChatBot instance."""
    bot = ChatBot(
        "ChatterBot",
        storage_adapter="chatterbot.storage.SQLStorageAdapter",
        database_uri=f"sqlite:///{DB_PATH}",
        logic_adapters=[
            {
                "import_path": "chatterbot.logic.BestMatch",
                "default_response": "I'm not sure I understand. Could you rephrase that?",
                "maximum_similarity_threshold": 0.90,
            }
        ],
        read_only=False,
    )
    return bot


def train_bot(bot: ChatBot, db_exists: bool) -> None:
    """Train the bot (skipped if the database already existed)."""

    print(c("\n  [*] Initialising ChatterBot...", "yellow"))

    if db_exists:
        print(c("  [OK] Database found — skipping training (using saved knowledge).", "green"))
        return

    print(c("  [*] First run — training on English corpus. Please wait...", "cyan"))

    # --- Corpus training -------------------------------------------------------
    corpus_trainer = ChatterBotCorpusTrainer(bot)
    corpora = [
        "chatterbot.corpus.english.greetings",
        "chatterbot.corpus.english.conversations",
        "chatterbot.corpus.english.humor",
        "chatterbot.corpus.english.trivia",
        "chatterbot.corpus.english.food",
        "chatterbot.corpus.english.science",
        "chatterbot.corpus.english.sports",
        "chatterbot.corpus.english.computers",
    ]
    for corpus in corpora:
        try:
            corpus_trainer.train(corpus)
            name = corpus.split(".")[-1]
            print(c(f"    [+] Trained: {name}", "green"))
        except Exception:
            pass  # Skip unavailable corpora silently

    # --- Custom conversation training ------------------------------------------
    list_trainer = ListTrainer(bot)
    list_trainer.train(CUSTOM_CONVERSATIONS)
    print(c("  [+] Trained on custom conversation data.", "green"))
    print(c("  [OK] Training complete!\n", "green", bold=True))


# ─────────────────────────────────────────────────────────────────────────────
#  Main chat loop
# ─────────────────────────────────────────────────────────────────────────────
def chat_loop(bot: ChatBot) -> None:
    """Run the interactive conversation loop."""
    history: list = []

    print(c("\n  [CHAT] Session started. Type 'help' for commands.\n", "green", bold=True))
    print(c("  " + "=" * 55, "blue"))

    while True:
        try:
            user_input = input(c("\n  You  >> ", "cyan", bold=True)).strip()
        except (KeyboardInterrupt, EOFError):
            print(c("\n\n  Goodbye! See you next time!\n", "yellow", bold=True))
            break

        if not user_input:
            print(c("  [!] Please type something!", "yellow"))
            continue

        lower = user_input.lower()

        # ── Built-in commands ──────────────────────────────────────────────────
        if lower == "help":
            print_help()
            continue

        if lower == "about":
            print_about()
            continue

        if lower == "clear":
            os.system("cls" if os.name == "nt" else "clear")
            print_banner()
            continue

        if lower == "history":
            if not history:
                print(c("\n  [i] No conversation history yet.\n", "yellow"))
            else:
                print(c(f"\n  [ Session History — {len(history)} exchange(s) ]", "yellow", bold=True))
                print(c("  " + "-" * 45, "blue"))
                for i, entry in enumerate(history, 1):
                    print(f"  {c(str(i).rjust(2), 'blue')}. {c('You:', 'cyan')} {entry['user']}")
                    print(f"      {c('Bot:', 'magenta')} {entry['bot']}")
                print()
            continue

        if lower in EXIT_TRIGGERS:
            print(c("\n  Thanks for chatting! Goodbye!\n", "yellow", bold=True))
            break

        # ── Get bot response ───────────────────────────────────────────────────
        try:
            response = bot.get_response(user_input)
            bot_reply = str(response)
        except Exception as e:
            bot_reply = "Oops — something went wrong on my end. Let's try again."
            print(c(f"  [ERR] {e}", "red"))

        # ── Display response ───────────────────────────────────────────────────
        timestamp = datetime.datetime.now().strftime("%H:%M")
        print(c(f"\n  Bot  >> ", "magenta", bold=True) + bot_reply + c(f"  [{timestamp}]", "blue"))

        history.append({"user": user_input, "bot": bot_reply})


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    os.system("cls" if os.name == "nt" else "clear")
    print_banner()

    db_exists = os.path.exists(DB_PATH)

    bot  = create_bot()
    train_bot(bot, db_exists)
    chat_loop(bot)


if __name__ == "__main__":
    main()
