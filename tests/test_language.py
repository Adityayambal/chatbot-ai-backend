from app.content.language import medium_to_lang_key, pick_lang


def test_medium_to_lang_key_known_and_case_insensitive():
    assert medium_to_lang_key("English") == "en"
    assert medium_to_lang_key("SEMI ENGLISH") == "en"
    assert medium_to_lang_key("marathi") == "mr"
    assert medium_to_lang_key("Gujarati") == "gu"


def test_medium_to_lang_key_unknown_falls_back_to_en():
    assert medium_to_lang_key("Klingon") == "en"


def test_pick_lang_prefers_requested_key():
    assert pick_lang({"mr": "namaste", "en": "hello"}, "mr") == "namaste"


def test_pick_lang_falls_back_to_en():
    assert pick_lang({"en": "hello"}, "mr") == "hello"


def test_pick_lang_falls_back_to_first_available():
    assert pick_lang({"gu": "x"}, "mr") == "x"


def test_pick_lang_empty_dict():
    assert pick_lang({}, "mr") == ""
