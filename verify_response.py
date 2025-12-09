import json

with open("response.json") as f:
    data = json.load(f)

print("=== Answer ===")
print(data["answer_ko"])

print("\n=== Citations ===")
for i, cit in enumerate(data["citations"]):
    print(f"[{i+1}] Doc: {cit['doc_id']}")
    print(f"    Page: {cit.get('page')}")
    print(f"    Ref: {cit['ref']}")
    anchor_snippet = cit.get('anchor', '')[:100].replace('\n', ' ')
    print(f"    Snippet: {anchor_snippet}...")
