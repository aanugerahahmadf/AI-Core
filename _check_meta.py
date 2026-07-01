import json
meta = json.load(open('data/metadata.json', 'r', encoding='utf-8'))
images = meta.get('images', [])
print('Total entries in metadata.json:', len(images))
for img in images:
    m = img.get('metadata', {})
    oid = m.get("owner_id", "?")
    print(f'  #{img["id"]:2d}: {str(m.get("type","?")):7s} | owner_id={str(oid):4s} | {m.get("name","?")}')
