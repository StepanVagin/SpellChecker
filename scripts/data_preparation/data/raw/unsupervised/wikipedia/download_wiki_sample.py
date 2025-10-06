import urllib.request
import json

articles = [
    "Python_(programming_language)",
    "Machine_learning",
    "Natural_language_processing",
    "Artificial_intelligence",
    "Data_science"
]

wiki_texts = []

for article_title in articles:
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{article_title}"
        req = urllib.request.Request(
            url,
            headers={
                'User-Agent': 'SpellCheckerBot/1.0 (Educational Project)',
                'Accept': 'application/json'
            }
        )
        
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read())
            title = data.get("title", article_title.replace("_", " "))
            extract = data.get("extract", "")
            
            wiki_text = f'<doc id="{len(wiki_texts)+1}" title="{title}">\n{extract}\n</doc>'
            wiki_texts.append(wiki_text)
            print(f"  ✓ Downloaded: {title}")
    
    except Exception as e:
        print(f"  ✗ Failed: {article_title}: {e}")

if wiki_texts:
    with open("extracted/AA/wiki_00", "w", encoding="utf-8") as f:
        f.write("\n".join(wiki_texts))
    print(f"\n✅ Downloaded {len(wiki_texts)} Wikipedia articles")
else:
    print("\n❌ Failed to download any Wikipedia articles")
    exit(1)
