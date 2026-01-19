import re
from typing import List

class AdvancedTextPreprocessor:
    """
    Advanced text preprocessing with evasion handling.

    Handles common evasion techniques:
    - Abbreviations (kys → kill yourself)
    - Leetspeak (tr4sh → trash)
    - Unicode tricks (Cyrillic → Latin)
    - Spacing evasion (k y s → kys)
    """

    def __init__(self, remove_stopwords=False):
        """
        Initialize advanced preprocessor.

        Args:
            remove_stopwords: Whether to remove common stopwords
        """
        self.remove_stopwords = remove_stopwords

        # Toxic abbreviations (gaming-specific)
        self.abbreviations = {
            'kys': 'kill yourself',
            'kmys': 'kill myself',
            'kms': 'kill myself',
            'gtfo': 'get the fuck out',
            'stfu': 'shut the fuck up',
            'gfy': 'go fuck yourself',
            'fys': 'fuck yourself',
            'foff': 'fuck off',
            'pos': 'piece of shit',
            'sob': 'son of a bitch',
            'mfer': 'motherfucker',
            'mofo': 'motherfucker',
            'af': 'as fuck',
            'ez': 'easy noob',  # Toxic in gaming context
            'gg ez': 'good game easy noob',
            'l2p': 'learn to play',
            'git gud': 'get good',
            'rekt': 'wrecked',
            'pwned': 'owned',
            'noob': 'newbie',
            'scrub': 'bad player',
        }

        # Leetspeak character mappings
        self.leetspeak = {
            '4': 'a', '@': 'a',
            '3': 'e', '€': 'e',
            '1': 'i', '!': 'i', '|': 'i',
            '0': 'o',
            '$': 's', '5': 's',
            '7': 't', '+': 't',
            '8': 'b',
            '6': 'g', '9': 'g',
        }

        # Unicode normalization (lookalike characters)
        self.unicode_map = {
            # Cyrillic → Latin
            'а': 'a', 'А': 'A', 'е': 'e', 'Е': 'E',
            'о': 'o', 'О': 'O', 'р': 'p', 'Р': 'P',
            'с': 'c', 'С': 'C', 'у': 'y', 'У': 'Y',
            'х': 'x', 'Х': 'X', 'і': 'i', 'І': 'I',
            'к': 'k', 'К': 'K', 'н': 'h', 'Н': 'H',
            'т': 't', 'Т': 'T', 'м': 'm', 'М': 'M',
            # Greek → Latin
            'α': 'a', 'Α': 'A', 'ε': 'e', 'Ε': 'E',
            'ο': 'o', 'Ο': 'O', 'ρ': 'p', 'Ρ': 'P',
            'β': 'b', 'Β': 'B', 'γ': 'g', 'Γ': 'G',
            'δ': 'd', 'Δ': 'D', 'ι': 'i', 'Ι': 'I',
            'κ': 'k', 'Κ': 'K', 'ν': 'n', 'Ν': 'N',
            'τ': 't', 'Τ': 'T', 'υ': 'u', 'Υ': 'U',
        }

        # Stopwords (minimal set)
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with'
        }

    def normalize_unicode(self, text: str) -> str:
        """Convert lookalike Unicode characters to ASCII."""
        for foreign, latin in self.unicode_map.items():
            text = text.replace(foreign, latin)
        return text

    def normalize_leetspeak(self, text: str) -> str:
        """Convert leetspeak to normal characters."""
        # Only replace if surrounded by letters (avoid false positives)
        normalized = text
        for leet, normal in self.leetspeak.items():
            # Replace leetspeak characters
            normalized = normalized.replace(leet, normal)
        return normalized

    def remove_spacing_evasion(self, text: str) -> str:
        """Detect and fix intentional spacing (k y s → kys)."""
        words = text.split()

        # Count single-character words
        single_chars = [w for w in words if len(w) == 1 and w.isalpha()]

        # If 60%+ are single characters, probably spacing evasion
        if len(words) > 0 and len(single_chars) >= len(words) * 0.6:
            return ''.join(words)

        return text

    def expand_abbreviations(self, text: str) -> str:
        """Expand toxic abbreviations."""
        words = text.split()
        expanded = []

        i = 0
        while i < len(words):
            # Check for multi-word abbreviations
            if i < len(words) - 1:
                two_word = ' '.join(words[i:i+2]).lower()
                if two_word in self.abbreviations:
                    expanded.append(self.abbreviations[two_word])
                    i += 2
                    continue

            # Check single word
            word = words[i]
            if word.lower() in self.abbreviations:
                expanded.append(self.abbreviations[word.lower()])
            else:
                expanded.append(word)
            i += 1

        return ' '.join(expanded)

    def clean_text(self, text: str) -> str:
        """
        Complete preprocessing pipeline with evasion handling.

        Args:
            text: Raw text message

        Returns:
            Cleaned and normalized text
        """
        if not isinstance(text, str):
            return ""

        # 1. Normalize Unicode lookalikes (Cyrillic/Greek → Latin)
        text = self.normalize_unicode(text)

        # 2. Lowercase
        text = text.lower()

        # 3. Remove URLs and mentions
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+', '', text)

        # 4. Keep only letters, numbers, spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)

        # 5. Fix spacing evasion (k y s → kys)
        text = self.remove_spacing_evasion(text)

        # 6. Normalize leetspeak (tr4sh → trash, n00b → noob)
        text = self.normalize_leetspeak(text)

        # 7. Expand abbreviations (kys → kill yourself)
        text = self.expand_abbreviations(text)

        # 8. Remove stopwords if enabled
        if self.remove_stopwords:
            words = text.split()
            text = ' '.join([w for w in words if w not in self.stopwords])

        # 9. Remove extra whitespace
        text = ' '.join(text.split())

        return text.strip()

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess multiple texts.

        Args:
            texts: List of raw text messages

        Returns:
            List of cleaned texts
        """
        return [self.clean_text(text) for text in texts]

    def show_example(self, text: str) -> None:
        """
        Show before/after preprocessing example.

        Args:
            text: Raw text to demonstrate preprocessing
        """
        cleaned = self.clean_text(text)
        print(f"Original: {text}")
        print(f"Cleaned:  {cleaned}")
        print()


def demo_advanced_preprocessing():
    """Demonstrate advanced preprocessing on evasion attempts."""

    examples = [
        # Abbreviations
        "kys noob",
        "gtfo trash",
        "stfu you pos",

        # Leetspeak
        "you're tr4sh",
        "n00b",
        "ur g@rbage",
        "fvck you",

        # Spacing evasion
        "k y s",
        "t r a s h",
        "n o o b",

        # Unicode tricks (Cyrillic)
        "you're trаsh",  # Cyrillic 'а'
        "nооb",          # Cyrillic 'о'

        # Mixed techniques
        "kys n00b",
        "ur tr4sh n00b",
        "g t f o",
        "u r p0s",

        # Should still detect
        "kill yourself noob",
        "you're trash",
    ]

    print("=" * 70)
    print("ADVANCED PREPROCESSING - EVASION HANDLING")
    print("=" * 70)
    print()
    print("Configuration: Abbreviation expansion + Leetspeak normalization")
    print("-" * 70)

    preprocessor = AdvancedTextPreprocessor(remove_stopwords=False)

    for example in examples:
        preprocessor.show_example(example)


if __name__ == "__main__":
    demo_advanced_preprocessing()
