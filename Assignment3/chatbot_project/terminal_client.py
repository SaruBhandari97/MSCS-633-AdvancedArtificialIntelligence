"""
terminal_client.py - Command-line chat interface for the ChatterBot Q&A bot.

Run this script from the project root to start an interactive terminal session:

    python terminal_client.py

Type 'quit' or 'exit' (case-insensitive) to end the session.

Author: Saru Bhandari
Course: MSCS-633-A01 Advanced Artificial Intelligence
Assignment: Hands-On Assignment 3
"""

import sys
import os

# ── Django setup ────────────────────────────────────────────────────────────
# Django must be configured before importing any project module, even when
# running outside of a web server context.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "chatbot_config.settings")

import django
django.setup()
# ────────────────────────────────────────────────────────────────────────────

from chatbot.bot import create_bot  # noqa: E402  (import after django.setup)


def run_chat() -> None:
    """
    Start the interactive terminal chat loop.

    Continuously prompts the user for input, passes it to the bot,
    and prints the bot's response. The loop ends when the user types
    'quit' or 'exit', or when EOF is received (Ctrl-D / Ctrl-Z).
    """
    # Initialise and train the bot
    bot = create_bot()

    print("=" * 55)
    print("  Welcome to CourseBot — your AI conversation partner!")
    print("  Type 'quit' or 'exit' to end the session.")
    print("=" * 55 + "\n")

    while True:
        try:
            # Read user input from stdin
            user_input = input("user: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Handle Ctrl-D / Ctrl-C gracefully
            print("\nbot: Goodbye! Have a great day!")
            sys.exit(0)

        # Allow the user to exit the loop cleanly
        if user_input.lower() in ("quit", "exit"):
            print("bot: Goodbye! Have a great day!")
            break

        # Skip blank input without querying the bot
        if not user_input:
            continue

        # Generate and display the bot's response
        response = bot.get_response(user_input)
        print(f"bot: {response}\n")


if __name__ == "__main__":
    run_chat()
