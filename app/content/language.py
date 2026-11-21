MEDIUM_TO_LANG: dict[str, str] = {
    "english": "en",
    "semi english": "en",
    "hindi": "hi",
    "marathi": "mr",
    "gujarati": "gu",
    "urdu": "ur",
    "kannada": "kn",
    "telugu": "te",
    "tamil": "ta",
    "bengali": "bn",
}


def medium_to_lang_key(medium_name: str) -> str:
    return MEDIUM_TO_LANG.get(medium_name.strip().lower(), "en")


def pick_lang(d: dict[str, str], lang_key: str) -> str:
    """Pick the localized value for lang_key, falling back to 'en', then the first key."""
    if not d:
        return ""
    if lang_key in d and d[lang_key]:
        return d[lang_key]
    if "en" in d and d["en"]:
        return d["en"]
    for value in d.values():
        if value:
            return value
    return ""
