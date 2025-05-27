import re
import unicodedata

# --- Amharic Specific Normalization Mappings ---
AMHARIC_NORMALIZATION_MAP = {
    '\u1200': '\u1200', '\u1201': '\u1201', '\u1202': '\u1202', '\u1203': '\u1203', '\u1204': '\u1204', '\u1205': '\u1205', '\u1206': '\u1206',
    '\u1240': '\u1200', '\u1241': '\u1201', '\u1242': '\u1202', '\u1243': '\u1203', '\u1244': '\u1204', '\u1245': '\u1205', '\u1246': '\u1206',
    '\u1280': '\u1200', '\u1281': '\u1201', '\u1282': '\u1202', '\u1283': '\u1203', '\u1284': '\u1204', '\u1285': '\u1205', '\u1286': '\u1206',

    '\u1208': '\u1208', '\u1209': '\u1209', '\u120A': '\u120A', '\u120B': '\u120B', '\u120C': '\u120C', '\u120D': '\u120D', '\u120E': '\u120E',

    '\u1220': '\u1220', '\u1221': '\u1221', '\u1222': '\u1222', '\u1223': '\u1223', '\u1224': '\u1224', '\u1225': '\u1225', '\u1226': '\u1226',
    '\u1230': '\u1220', '\u1231': '\u1221', '\u1232': '\u1222', '\u1233': '\u1223', '\u1234': '\u1224', '\u1235': '\u1225', '\u1236': '\u1226',

    '\u12A0': '\u12A0', '\u12A1': '\u12A1', '\u12A2': '\u12A2', '\u12A3': '\u12A3', '\u12A4': '\u12A4', '\u12A5': '\u12A5', '\u12A6': '\u12A6',
    '\u12D0': '\u12A0', '\u12D1': '\u12A1', '\u12D2': '\u12A2', '\u12D3': '\u12A3', '\u12D4': '\u12A4', '\u12D5': '\u12A5', '\u12D6': '\u12A6',

    '\u12D8': '\u12D8', '\u12D9': '\u12D9', '\u12DA': '\u12DA', '\u12DB': '\u12DB', '\u12DC': '\u12DC', '\u12DD': '\u12DD', '\u12DE': '\u12DE',
    '\u1350': '\u12D8', '\u1351': '\u12D9', '\u1352': '\u12DA', '\u1353': '\u12DB', '\u1354': '\u12DC', '\u1355': '\u12DD', '\u1356': '\u12DE',

    # Punctuation Standardization (Amharic specific punctuation to ASCII equivalents)
    '\u1361': ',', # Ethiopian Comma (፣)
    '\u1362': '.', # Ethiopian Full Stop (።)
    '\u1363': ';', # Ethiopian Semicolon (፤)
    '\u1364': ':', # Ethiopian Colon (፥)
    '\u1365': '?', # Ethiopian Question Mark (፦)
    '\u1366': '|', # Ethiopian Paragraph Separator (፧) - often treated as line break or removed
    '\u1367': '!', # Ethiopian Exclamation Mark (፨)
    '\u1368': ' ', # Ethiopian Word Space (ᎈ) - handled by general whitespace later

    # Convert commonly used non-Amharic digits to ASCII
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4', '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9', # Arabic digits
    '߀': '0', '߁': '1', '߂': '2', '߃': '3', '߄': '4', '߅': '5', '߆': '6', '߇': '7', '߈': '8', '߉': '9', # N'Ko digits
    '༠': '0', '༡': '1', '༢': '2', '༣': '3', '༤': '4', '༥': '5', '༦': '6', '༧': '7', '༨': '8', '༩': '9', # Tibetan digits
}

# --- More Comprehensive Amharic Stopwords ---
AMHARIC_STOPWORDS = {
    "እና", "ነው", "ላይ", "የ", "ለ", "በ", "ከ", "ወደ", "ይህ", "ያ", "እነዚህ", "እነዚያ", "እኔ", "አንተ",
    "እሱ", "እሷ", "እኛ", "እናንተ", "እነሱ", "ነበር", "ሆነ", "ነው", "ብቻ", "አሁን", "በጣም", "አለ", "የለም",
    "እንዲሁም", "ስለ", "ለዚህ", "ምንም", "ያለ", "ምን", "እንዴት", "የት", "መቼ", "ለምን", "ማን", "የትኛው",
    "እንኳን", "ብዙ", "ሌላ", "አንድ", "ሁለት", "ሶስት", "አራት", "አምስት", "ስድስት", "ሰባት", "ስምንት", "ዘጠኝ", "አስር",
    "ሃያ", "ሰላሳ", "አርባ", "ሃምሳ", "ስልሳ", "ሰባ", "ሰማኒያ", "ዘጠና", "መቶ", "ሺህ", "ሚሊዮን", "ቢሊዮን",
    "አሁን", "ትናንት", "ዛሬ", "ነገ", "አንዴ", "ሁልጊዜ", "አንዳንድ", "የሆነ", "ሁሉም", "ወደ", "ከ", "ውስጥ", "ውጪ",
    "በኋላ", "በፊት", "በላይ", "በታች", "እስከ", "እስከዚህ", "አሁን", "ቀን", "ሳምንት", "ወር", "ዓመት", "ጊዜ", "ደቂቃ", "ሰዓት",
    "እጅግ", "አጥብቆ", "ፈጽሞ", "በጣም", "በደንብ", "ብቻ", "ብቻም", "ተመሳሳይ", "ልዩ", "አዲስ", "አሮጌ", "ትልቅ", "ትንሽ",
    "ጥሩ", "መጥፎ", "አፍ", "ዓይን", "እግር", "እጅ", "ሰው", "ሴት", "ወንድ", "ህፃን", "ልጅ", "እናቴ", "አባቴ", "ወንድሜ", "እህቴ",
    "ሀገር", "ከተማ", "ቤት", "መንገድ", "ዛፍ", "ውሃ", "ምግብ", "መጽሐፍ", "ጠረጴዛ", "ወንበር", "በር", "መስኮት", "ስም", "ቃል",
    "አሁን", "ነገ", "ዛሬ", "ትናንት", "መቼም", "የትም", "እንዴትም", "መጥፎም", "ጥሩም", "እየ", "ማለት", "ም", "የሚ", "አም",
    "አሉት", "አለች", "አለ", "ነበሩ", "ነበርን", "ነበራችሁ", "ነበራችሁ", "ነበረ", "ነበሩ"
}

# --- Regex Patterns for Efficient Preprocessing ---
URL_PATTERN = re.compile(r'http\S+|www\S+|https\S+')
MENTION_HASHTAG_PATTERN = re.compile(r'@\w+|#\w+')

EMOJI_SYMBOL_PATTERN = re.compile(
    '['
    '\U0001F600-\U0001F64F'  # emoticons
    '\U0001F300-\U0001F5FF'  # symbols & pictographs
    '\U0001F680-\U0001F6FF'  # transport & map symbols
    '\U0001F1E0-\U0001F1FF'  # flags (iOS)
    '\U00002702-\U000027B0'  # Dingbats
    '\U000024C2-\U0001F251'  # Enclosed CJK letters and months, etc.
    '\U0001F900-\U0001F9FF'  # Supplemental Symbols and Pictographs
    '\U0001FA00-\U0001FA6F'  # Chess Symbols, Symbols and Pictographs Extended-A
    '\U00002B00-\U00002BFF'  # Miscellaneous Symbols and Arrows
    '\U00002000-\U0000206F'  # General Punctuation (like zero-width spaces)
    '\uFE00-\uFE0F'          # Variation Selectors
    '\u0020-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E' # ASCII Punctuation
    '\u200C'                 # Zero Width Non-Joiner
    ']+', flags=re.UNICODE
)

CLEAN_TEXT_PATTERN = re.compile(r'[^\u1200-\u137F0-9\s]')
MULTIPLE_SPACE_PATTERN = re.compile(r'\s+')

# --- Amharic Sentence Tokenization Pattern ---
# Splits by common Amharic and English sentence-ending punctuation (., ?, !, ።)
# It captures the delimiter to keep it as part of the sentence or for later inspection.
# It handles multiple punctuation marks, e.g., "!!!"
# It doesn't split on periods within numbers (e.g., 10.5) or abbreviations (less common in Amharic)
# For simplicity, it assumes sentence boundaries are marked by these explicit punctuation.
AMHARIC_SENTENCE_DELIMITERS = re.compile(r'([።?!]|[.,;:])(?=\s+|\n|$)')


def normalize_amharic_chars(text):
    """Applies specific Amharic character normalization based on the map."""
    normalized_text = []
    for char in text:
        normalized_text.append(AMHARIC_NORMALIZATION_MAP.get(char, char))
    return "".join(normalized_text)

def preprocess_amharic_text(text):
    """
    Applies comprehensive preprocessing to Amharic text for NLP tasks.
    Handles URLs, mentions, hashtags, emojis, punctuation, common Amharic
    character variations, and stopwords.
    """
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize('NFKC', text)
    text = normalize_amharic_chars(text)
    text = URL_PATTERN.sub(r'', text)
    text = MENTION_HASHTAG_PATTERN.sub(r'', text)
    text = EMOJI_SYMBOL_PATTERN.sub(r' ', text)
    text = CLEAN_TEXT_PATTERN.sub(r' ', text)

    tokens = text.split()
    tokens = [word for word in tokens if word not in AMHARIC_STOPWORDS]

    cleaned_text = MULTIPLE_SPACE_PATTERN.sub(' ', " ".join(tokens)).strip()

    return cleaned_text

def tokenize_amharic_sentences(text):
    """
    Splits a given Amharic text into sentences.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    # Split by the defined delimiters, keeping the delimiters as part of the sentence
    sentences = []
    last_idx = 0
    for match in AMHARIC_SENTENCE_DELIMITERS.finditer(text):
        # Add the part before the delimiter
        sentence_part = text[last_idx:match.end()].strip()
        if sentence_part:
            sentences.append(sentence_part)
        last_idx = match.end()

    # Add any remaining text after the last delimiter
    remaining_text = text[last_idx:].strip()
    if remaining_text:
        sentences.append(remaining_text)

    # If no delimiters were found, the whole text is one sentence
    if not sentences and text.strip():
        sentences.append(text.strip())
        
    return sentences

# Example usage (for testing this module independently)
if __name__ == "__main__":
    test_texts = [
        "ይህ በጣም ጥሩ ጽሑፍ ነው! https://example.com #አማርኛ #ጥሩ",
        "እንዴት ነህ @abebew? የፖለቲካ ንግግር እና የጥላቻ ንግግር ልዩነት አለ።",
        "አንተ ደደብ ነህ:: ይህ የጥላቻ ንግግር ነው!!",
        "ይህ ጽሑፍ በዛሬው ዕለት ኅትመት ላይ ቀርቧል።",
        "😂 ሰላም ኁሉ! እናቴ እንዴት ነሽ?",
        "  ይህ    ብዙ ክፍተቶች አሉት።   ",
        "፨ በዛሬው ዕለት የኢትዮጵያ ዜና ታወቀ።",
        "እኔ 12345 ስልክ አለኝ። ٠٩٨٧٦٥٤٣٢١",
        "ምንድነው የሰሩት? 🤷‍♀️ እጅግ በጣም ጥሩ!",
        "አማርኛ። ይህ የሙከራ ጽሑፍ ነው። በብዙ ዓረፍተ ነገሮች አማርኛ እየጻፍኩ ነው።",
        "አንድ ብቻ ዓረፍተ ነገር."
    ]

    print("--- Testing Amharic Preprocessing ---")
    for i, text in enumerate(test_texts):
        print(f"\nOriginal {i+1}: {text}")
        processed_text = preprocess_amharic_text(text)
        print(f"Processed {i+1}: {processed_text}")

    print("\n--- Testing Amharic Sentence Tokenization ---")
    for i, text in enumerate(test_texts):
        print(f"\nOriginal Text {i+1}: {text}")
        sentences = tokenize_amharic_sentences(text)
        for j, sentence in enumerate(sentences):
            print(f"  Sentence {j+1}: '{sentence}'")

    print("\n--- Note on Amharic Normalization and Stopwords ---")
    print("The AMHARIC_NORMALIZATION_MAP and AMHARIC_STOPWORDS are expanded samples.")
    print("For industrial-grade Amharic NLP, dedicated linguistic resources or libraries")
    print("offering morphological analysis and more exhaustive normalization are ideal.")
    print("This implementation provides a solid foundation for a school project.")