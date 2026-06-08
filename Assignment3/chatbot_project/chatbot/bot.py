"""
bot.py - ChatterBot initialization and training module.

This module sets up and trains the ChatterBot instance using both the
built-in English corpus and a custom Q&A dataset covering artificial
intelligence, Python programming, and general computer science topics.

Author: Saru Bhandari
Course: MSCS-633-A01 Advanced Artificial Intelligence
Assignment: Hands-On Assignment 3
"""

import logging
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer, ListTrainer

# Suppress noisy chatterbot training output for cleaner console UX
logging.getLogger("chatterbot").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Custom training data
# Each pair is [user_input, bot_response]. The ListTrainer feeds these as
# conversation sequences so the bot learns domain-specific Q&A.
# ---------------------------------------------------------------------------
CUSTOM_CONVERSATIONS = [

    # -- Greetings & small talk ---------------------------------------------
    ["Good morning! How are you doing?",
     "Good morning! I am doing very well, thank you for asking. How can I help you today?"],

    ["Hello, how are you?",
     "Hello! I am doing great. What can I assist you with?"],

    ["How are you?",
     "I am functioning perfectly, thank you! What would you like to talk about?"],

    ["What is your name?",
     "My name is CourseBot. I am an AI-powered chatbot built with ChatterBot and Django."],

    ["Who created you?",
     "I was created by Ganesh Prasad Bhandari as part of MSCS-633-A01 Advanced Artificial Intelligence at the University of the Cumberlands."],

    ["Thank you!",
     "You are welcome! Feel free to ask me anything else."],

    ["Thanks",
     "Happy to help! Is there anything else you would like to know?"],

    ["Goodbye",
     "Goodbye! Have a wonderful day. Type 'exit' anytime to end our session."],

    # -- Artificial Intelligence --------------------------------------------
    ["What is artificial intelligence?",
     "Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems. It includes learning, reasoning, problem-solving, perception, and language understanding."],

    ["Tell me about artificial intelligence.",
     "Artificial Intelligence is a broad field of computer science focused on building smart machines capable of performing tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation."],

    ["What are the types of artificial intelligence?",
     "There are three main types of AI: Narrow AI (designed for a specific task, like voice assistants), General AI (human-level intelligence across many tasks), and Super AI (hypothetical intelligence surpassing human capability)."],

    ["What is machine learning?",
     "Machine Learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves."],

    ["What is deep learning?",
     "Deep Learning is a subset of machine learning that uses neural networks with many layers (deep neural networks) to analyze various factors of data. It powers applications like image recognition, speech recognition, and natural language processing."],

    ["What is natural language processing?",
     "Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language. It powers chatbots, translation tools, sentiment analysis, and voice assistants."],

    ["What is a neural network?",
     "A neural network is a series of algorithms that attempts to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. It consists of layers of interconnected nodes or neurons."],

    ["What is the difference between AI and machine learning?",
     "AI is the broader concept of machines being able to carry out tasks in a smart way. Machine Learning is a specific application of AI that gives machines the ability to learn from data without being explicitly programmed."],

    # -- Python Programming -------------------------------------------------
    ["Tell me about Python programming.",
     "Python is a high-level, interpreted, general-purpose programming language known for its simplicity and readability. It is widely used in web development, data science, artificial intelligence, machine learning, automation, and scientific computing."],

    ["What is Python?",
     "Python is a versatile, beginner-friendly programming language created by Guido van Rossum in 1991. Its clean syntax and extensive libraries make it one of the most popular languages in AI, data science, and web development."],

    ["Why is Python popular for AI?",
     "Python is popular for AI because of its simple syntax, extensive libraries such as TensorFlow, PyTorch, scikit-learn, and NumPy, a large community, and strong support for data manipulation and visualization."],

    ["What are Python libraries used in AI?",
     "Key Python libraries used in AI and machine learning include TensorFlow, PyTorch, scikit-learn, Keras, NumPy, Pandas, Matplotlib, NLTK, and SpaCy."],

    ["What is Django?",
     "Django is a high-level Python web framework that encourages rapid development and clean, pragmatic design. It follows the Model-View-Template (MVT) pattern and includes built-in features for authentication, database management, and URL routing."],

    # -- ChatterBot ---------------------------------------------------------
    ["What is ChatterBot?",
     "ChatterBot is a Python library that makes it easy to generate automated responses to a user's input. It uses machine learning algorithms to select the best response from a set of known conversations it has been trained on."],

    ["How does ChatterBot work?",
     "ChatterBot works by training on a dataset of conversation pairs. When a user sends a message, it computes similarity scores between the input and known statements, then returns the response with the highest confidence score."],

    ["What corpus does this bot use?",
     "This bot is trained on ChatterBot's built-in English corpus, which covers greetings, general conversation, and trivia. It is also trained on a custom dataset covering AI, Python, and computer science topics."],

    # -- Computer Science ---------------------------------------------------
    ["What is a database?",
     "A database is an organized collection of structured data stored electronically. Common types include relational databases (like SQLite, PostgreSQL, MySQL) and NoSQL databases (like MongoDB, Redis)."],

    ["What is an algorithm?",
     "An algorithm is a step-by-step procedure or set of rules designed to perform a specific task or solve a problem. Algorithms are the foundation of all computer programs."],

    ["What is object-oriented programming?",
     "Object-Oriented Programming (OOP) is a programming paradigm based on the concept of objects, which contain data (attributes) and code (methods). Key principles include encapsulation, inheritance, polymorphism, and abstraction."],

    ["What is a API?",
     "An API (Application Programming Interface) is a set of rules and protocols that allows different software applications to communicate with each other. REST APIs are commonly used in web development."],
]


def create_bot(bot_name: str = "CourseBot") -> ChatBot:
    """
    Create, configure, and train a ChatterBot instance.

    Trains on two sources:
      1. ChatterBot's built-in English corpus (greetings, general conversation).
      2. A custom Q&A list covering AI, Python, Django, and computer science.

    Args:
        bot_name: Display name for the chatbot (default: "CourseBot").

    Returns:
        A trained ChatBot instance ready to generate responses.
    """
    bot = ChatBot(
        bot_name,
        # SQLite storage persists trained data between sessions
        storage_adapter="chatterbot.storage.SQLStorageAdapter",
        database_uri="sqlite:///chatbot.sqlite3",
        # BestMatch selects the highest-confidence response
        logic_adapters=[
            {
                "import_path": "chatterbot.logic.BestMatch",
                # Return this message when no match exceeds the threshold
                "default_response": (
                    "I'm sorry, I don't have a good answer for that. "
                    "Try asking about artificial intelligence, Python, "
                    "machine learning, or Django!"
                ),
                "maximum_similarity_threshold": 0.75,
            }
        ],
        read_only=False,
    )

    # --- Train on built-in English corpus ---
    corpus_trainer = ChatterBotCorpusTrainer(bot)
    print("[INFO] Training the bot on the English corpus — please wait...")
    corpus_trainer.train("chatterbot.corpus.english")

    # --- Train on custom Q&A conversations ---
    list_trainer = ListTrainer(bot)
    print("[INFO] Training the bot on custom AI & Python Q&A data...")
    for conversation in CUSTOM_CONVERSATIONS:
        list_trainer.train(conversation)

    print("[INFO] Training complete. You can start chatting!\n")

    return bot
