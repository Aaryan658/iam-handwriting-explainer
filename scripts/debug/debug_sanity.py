import re

def sanity_check(text):
    words = text.split()
    if len(words) < 3:
        return True
        
    # Check if mostly single characters or digits
    single_char_or_digit_words = sum(1 for w in words if len(w) == 1 or w.isdigit())
    if single_char_or_digit_words / len(words) >= 0.5:
        return True
        
    # Check alphabetic content ratio
    letters = sum(c.isalpha() for c in text)
    non_whitespace = sum(not c.isspace() for c in text)
    if non_whitespace > 0 and letters / non_whitespace < 0.4:
        return True
        
    return False

tests = [
    "0 1",
    "put down a resolution on the subject",
    "and he is to be backed by Mr. Will",
    "1 2 3 a b c d",
    "a b c",
    "test 123",
    "Valid transcription text right here."
]

for t in tests:
    print(f"'{t}': {sanity_check(t)}")
