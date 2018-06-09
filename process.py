import re

link_pattern = r'\[\[([A-z ]*?)(?:\|([A-z ]*?))?\]\]'

# [[article link|hyperlink text]]
# match.group(0) returns whole thing
# match.group(1) returns article link
# match.group(2) returns hyperlink text

def return_links(text):
    results = []
    for match in re.finditer(link_pattern, text):
        results.append(match.group(1))
    return results
