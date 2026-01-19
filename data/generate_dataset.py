import random
import csv
import os

# Define message templates and components
non_toxic_messages = [
    "gg well played team!",
    "good game everyone",
    "nice shot!",
    "great teamwork guys",
    "thanks for the game",
    "let's try again",
    "we can do this",
    "nice try!",
    "almost had it",
    "good luck have fun",
    "glhf everyone",
    "wp wp",
    "that was close",
    "good defense",
    "nice play",
    "lets push together",
    "follow me",
    "i'll help you",
    "cover me please",
    "need backup",
    "watch out behind you",
    "nice kill",
    "enemy spotted",
    "let's regroup",
    "wait for team",
    "good call",
    "thanks for heal",
    "nice save",
    "that was fun",
    "rematch?",
    "add me as friend",
]

toxic_messages = [
    "you're trash",
    "uninstall the game noob",
    "kill yourself",
    "you suck so bad",
    "worst player ever",
    "delete this game idiot",
    "you're useless",
    "brain dead player",
    "go back to tutorial",
    "how are you this bad",
    "stop feeding moron",
    "you're throwing the game",
    "reported for being bad",
    "i hope you lose every game",
    "you don't deserve to play",
    "pathetic loser",
    "absolute garbage player",
    "you're a waste of space",
    "stupid noob",
    "get cancer",
    "die in a fire",
    "your family must be ashamed",
    "end your life",
    "you're retarded",
    "neck yourself",
    "kys noob",
    "trash team",
    "uninstall you monkey",
    "how are you still alive",
    "i hope you get banned",
]

# Mixed/ambiguous messages that depend on context
ambiguous_messages = [
    "i'll kill you",
    "you're dead",
    "destroyed you",
    "ez noob",
    "too easy",
    "sit down",
    "get rekt",
    "owned",
    "smashed you",
    "you got lucky",
    "camper noob",
    "stop camping",
    "hacker",
    "nice hacks",
    "reported",
    "cheater",
    "tryhard",
    "sweaty player",
    "why so serious",
    "it's just a game",
]

# Game context modifiers
game_contexts = [
    ("pvp_combat", 0.2),      # In-combat, aggressive language more acceptable
    ("team_chat", 0.5),       # Team coordination
    ("all_chat", 0.7),        # Public chat, higher toxicity risk
    ("post_game", 0.6),       # After match
    ("competitive", 0.8),     # Ranked/competitive mode, higher tension
    ("casual", 0.3),          # Casual mode, relaxed
]

def generate_dataset(num_samples=2000, output_file="raw/gaming_chat_dataset.csv"):
    """Generate synthetic gaming chat dataset with toxicity and context scores."""

    dataset = []

    # Generate non-toxic messages
    for _ in range(int(num_samples * 0.5)):
        message = random.choice(non_toxic_messages)
        toxicity = random.uniform(0.0, 0.2)  # Low toxicity
        context_type, base_context = random.choice(game_contexts)
        context_score = random.uniform(max(0, base_context - 0.2), min(1, base_context + 0.2))

        dataset.append({
            "message": message,
            "toxicity": round(toxicity, 3),
            "context_score": round(context_score, 3),
            "context_type": context_type,
            "label": 0  # Non-toxic
        })

    # Generate toxic messages
    for _ in range(int(num_samples * 0.35)):
        message = random.choice(toxic_messages)
        toxicity = random.uniform(0.7, 1.0)  # High toxicity
        context_type, base_context = random.choice(game_contexts)
        context_score = random.uniform(max(0, base_context - 0.2), min(1, base_context + 0.2))

        dataset.append({
            "message": message,
            "toxicity": round(toxicity, 3),
            "context_score": round(context_score, 3),
            "context_type": context_type,
            "label": 1  # Toxic
        })

    # Generate ambiguous messages (context-dependent)
    for _ in range(int(num_samples * 0.15)):
        message = random.choice(ambiguous_messages)
        context_type, base_context = random.choice(game_contexts)
        context_score = random.uniform(max(0, base_context - 0.2), min(1, base_context + 0.2))

        # Toxicity depends on context
        if context_type in ["pvp_combat", "casual"]:
            toxicity = random.uniform(0.2, 0.5)  # Less toxic in game context
            label = 0
        else:
            toxicity = random.uniform(0.5, 0.8)  # More toxic in other contexts
            label = 1

        dataset.append({
            "message": message,
            "toxicity": round(toxicity, 3),
            "context_score": round(context_score, 3),
            "context_type": context_type,
            "label": label
        })

    # Shuffle dataset
    random.shuffle(dataset)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write to CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['message', 'toxicity', 'context_score', 'context_type', 'label']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dataset)

    print(f"✓ Generated {len(dataset)} samples")
    print(f"✓ Saved to {output_file}")
    print(f"\nDataset distribution:")
    print(f"  Non-toxic: {sum(1 for d in dataset if d['label'] == 0)} ({sum(1 for d in dataset if d['label'] == 0)/len(dataset)*100:.1f}%)")
    print(f"  Toxic: {sum(1 for d in dataset if d['label'] == 1)} ({sum(1 for d in dataset if d['label'] == 1)/len(dataset)*100:.1f}%)")

    return dataset

if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    generate_dataset(num_samples=2000)
