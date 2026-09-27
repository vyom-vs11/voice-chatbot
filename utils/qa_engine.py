"""
Siri-Style Open Knowledge and Dynamic Skills Engine
- Open-Domain Knowledge Search via Wikipedia REST API
- Real-time Clock & Date Skills
- Mathematical & Percentage Calculations
- Siri-style Witty Easter Eggs & Conversational Persona
"""

import re
import datetime
import urllib.parse
import urllib.request
import json

# Siri-style conversational responses & Easter eggs
SIRI_PERSONA = {
    "who are you": "I am VoiceBot, your personal voice assistant inspired by Siri, powered by Deep Learning and Speech Recognition.",
    "what is your name": "You can call me VoiceBot. I'm your intelligent virtual assistant.",
    "hey siri": "Hello there! How can I help you today?",
    "are you siri": "I'm inspired by Siri, designed as a voice-enabled AI assistant running on deep learning neural networks.",
    "who created you": "I was designed as an advanced AI demonstration project combining Speech Recognition, Bi-LSTM Deep Learning, and Open Knowledge systems.",
    "do you have feelings": "I don't have human feelings, but I'm always happy to help you find answers and solve problems!",
    "what is the meaning of life": "42, according to The Hitchhiker's Guide to the Galaxy. But practically, it's about learning, connecting, and creating meaning.",
    "tell me a secret": "I don't have many secrets, but here's one: deep learning models are essentially complex linear algebra and calculus optimizing weights!",
    "are you smart": "I try my best! I can process your voice, classify intents with a neural network, and search millions of topics across world knowledge.",
    "beatbox": "Boots and cats and boots and cats and boots and cats! 🎧",
    "tell me a story": "Once upon a time, a user spoke into a microphone. An audio waveform was captured, transformed into text, processed through a neural network, and they chatted happily ever after.",
    "what can you do": "You can ask me almost anything! From operational hours, campus admissions, and support, to world history, sports like cricket and football, math calculations, or the current time."
}


def get_current_time_or_date(query: str) -> str:
    """Answers real-time questions about the current time, day, or date."""
    q = query.lower()
    now = datetime.datetime.now()

    time_patterns = [
        r"\bwhat\s+is\s+the\s+time\b",
        r"\bcurrent\s+time\b",
        r"\btime\s+now\b",
        r"\bwhat\s+time\s+is\s+it\b",
        r"\btell\s+me\s+the\s+time\b"
    ]
    for pattern in time_patterns:
        if re.search(pattern, q):
            return f"The current local time is {now.strftime('%I:%M %p')}."

    date_patterns = [
        r"\bwhat\s+day\s+is\s+today\b",
        r"\bwhat\s+is\s+today\'?s?\s+date\b",
        r"\btoday\'?s?\s+date\b",
        r"\bcurrent\s+date\b",
        r"\bwhich\s+day\b",
        r"\bwhat\s+year\s+is\s+this\b"
    ]
    for pattern in date_patterns:
        if re.search(pattern, q):
            return f"Today is {now.strftime('%A, %B %d, %Y')}."

    return None


def calculate_math_or_percent(query: str) -> str:
    """Evaluates mathematical expressions, arithmetic, and percentages."""
    q = query.lower().strip()

    # Percentage check: "what is 20 percent of 500"
    m_pct = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+(\d+(?:\.\d+)?)", q)
    if m_pct:
        pct = float(m_pct.group(1))
        total = float(m_pct.group(2))
        res = (pct / 100.0) * total
        if res.is_integer():
            res = int(res)
        return f"{pct}% of {int(total) if total.is_integer() else total} is {res}."

    # General math: "what is 25 * 4", "calculate 150 / 3"
    m_math = re.search(r"(?:what\s+is|calculate|evaluate)?\s*([0-9\.\s\+\-\*\/\(\)]+)(?:\?)?$", q)
    if m_math:
        expr = m_math.group(1).strip()
        if re.search(r"[\+\-\*\/]", expr) and re.search(r"\d", expr):
            if re.match(r"^[\d\.\s\+\-\*\/\(\)]+$", expr):
                try:
                    result = eval(expr, {"__builtins__": None}, {})
                    if isinstance(result, float) and result.is_integer():
                        result = int(result)
                    return f"The answer to {expr} is {result}."
                except Exception:
                    pass
    return None


def fetch_wikipedia_knowledge(search_term: str) -> str:
    """
    Queries the open Wikipedia REST API for concise 2-3 sentence conversational answers
    on any person, place, sports player, concept, event, or scientific phenomenon.
    """
    try:
        # 1. Clean query
        cleaned = re.sub(r"^(?:who\s+is|what\s+is|tell\s+me\s+about|where\s+is|define|who\s+was|what\s+was|explain|which\s+is|how\s+does)\s+", "", search_term, flags=re.IGNORECASE)
        # Strip leading article "the", "a", "an"
        cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"[^\w\s]", "", cleaned).strip()
        if not cleaned or len(cleaned) < 2:
            return None

        # Try cleaned term, then fallback to original search term
        candidates = [cleaned, search_term]
        canonical_title = None

        for term in candidates:
            q_enc = urllib.parse.quote(term)
            search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={q_enc}&limit=1&namespace=0&format=json"
            req = urllib.request.Request(search_url, headers={"User-Agent": "VoiceBotAI/2.0 (academic; student@project.edu)"})
            
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data and len(data) > 1 and data[1]:
                    canonical_title = data[1][0]
                    break

        if canonical_title:
            title_enc = urllib.parse.quote(canonical_title.replace(" ", "_"))
            sum_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title_enc}"
            req2 = urllib.request.Request(sum_url, headers={"User-Agent": "VoiceBotAI/2.0 (academic; student@project.edu)"})
            
            with urllib.request.urlopen(req2, timeout=4) as resp2:
                summary_data = json.loads(resp2.read().decode("utf-8"))
                extract = summary_data.get("extract")
                if extract:
                    sentences = re.split(r"(?<=[.!?])\s+", extract)
                    concise = " ".join(sentences[:3]) if len(sentences) >= 3 else extract
                    return concise
    except Exception as e:
        print(f"[Knowledge Search Error]: {e}")
    return None


def resolve_knowledge_query(query: str) -> dict:
    """
    Siri-style open-domain resolver:
    1. Real-time Clock / Calendar
    2. Mathematical Calculations & Percentages
    3. Siri Conversational Persona & Easter Eggs
    4. Open-Domain World Knowledge (Wikipedia)
    """
    q_lower = query.lower().strip()

    # 1. Time / Date
    time_ans = get_current_time_or_date(query)
    if time_ans:
        return {
            "answered": True,
            "response": time_ans,
            "intent": "siri_time",
            "confidence": 0.99,
            "source": "Apple Clock Skill"
        }

    # 2. Math & Percentage
    math_ans = calculate_math_or_percent(query)
    if math_ans:
        return {
            "answered": True,
            "response": math_ans,
            "intent": "siri_calculator",
            "confidence": 0.99,
            "source": "Calculator Skill"
        }

    # 3. Siri Persona & Witty Easter Eggs
    cleaned_q = re.sub(r"[^\w\s]", "", q_lower).strip()
    for trigger, resp in SIRI_PERSONA.items():
        if trigger in cleaned_q or cleaned_q in trigger:
            return {
                "answered": True,
                "response": resp,
                "intent": "siri_chitchat",
                "confidence": 0.98,
                "source": "Siri Dialogue Engine"
            }

    # 4. Open-Domain Knowledge Search (Cricket, Football, People, Places, Science)
    knowledge_patterns = [
        r"\bwho\s+is\b",
        r"\bwhat\s+is\b",
        r"\btell\s+me\s+about\b",
        r"\bwhere\s+is\b",
        r"\bdefine\b",
        r"\bwho\s+was\b",
        r"\bwhat\s+was\b",
        r"\bhow\s+does\b",
        r"\bwhy\s+is\b",
        r"\bcapital\s+of\b",
        r"\bcricketer\b",
        r"\bcricket\b",
        r"\bfootball\b",
        r"\bplayer\b"
    ]
    is_knowledge_query = any(re.search(pat, q_lower) for pat in knowledge_patterns)

    # Exclude assistant specific domain questions from being sent to Wikipedia
    domain_exclusions = ["working hours", "campus location", "your name", "who are you", "what can you do"]
    if any(ex in q_lower for ex in domain_exclusions):
        is_knowledge_query = False

    if is_knowledge_query:
        if "best cricketer" in q_lower or "greatest cricketer" in q_lower:
            return {
                "answered": True,
                "response": (
                    "In cricket history, Sir Donald Bradman is widely considered the greatest batsman of all time with a Test average of 99.94. "
                    "In modern cricket, Sachin Tendulkar and Virat Kohli are universally celebrated as among the greatest of all time across all formats."
                ),
                "intent": "siri_knowledge",
                "confidence": 0.98,
                "source": "Siri Sports Knowledge"
            }

        wiki_summary = fetch_wikipedia_knowledge(query)
        if wiki_summary:
            return {
                "answered": True,
                "response": wiki_summary,
                "intent": "siri_knowledge",
                "confidence": 0.96,
                "source": "Wikipedia Knowledge Base"
            }

    return {"answered": False, "response": None, "intent": None, "confidence": 0.0, "source": None}
