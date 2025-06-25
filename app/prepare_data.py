import json
import re

# قراءة qa.jsonl
qa_jsonl = []
with open('data/qa.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        try:
            item = json.loads(line.strip())
            if 'question' in item and 'answer' in item:
                qa_jsonl.append({
                    'type': 'qa',
                    'question': item['question'].strip(),
                    'answer': item['answer'].strip()
                })
        except Exception:
            continue

# قراءة qa.txt
qa_txt = []
with open('data/qa.txt', 'r', encoding='utf-8') as f:
    content = f.read()
    # تقسيم الأسئلة والأجوبة
    qa_pairs = re.split(r'Q: ', content)
    for pair in qa_pairs:
        if not pair.strip():
            continue
        parts = pair.split('A:')
        if len(parts) == 2:
            question = parts[0].replace('\n', ' ').strip()
            answer = parts[1].replace('\n', ' ').strip()
            qa_txt.append({
                'type': 'qa',
                'question': question,
                'answer': answer
            })

# قراءة exercises.txt
exercises = []
with open('data/exercises.txt', 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
    for line in lines:
        if not line.strip():
            continue
        # اسم التمرين والوصف
        if '–' in line:
            name, desc = line.split('–', 1)
            exercises.append({
                'type': 'exercise',
                'name': name.strip(),
                'description': desc.strip()
            })

# دمج كل البيانات
all_data = qa_jsonl + qa_txt + exercises

# حفظ في ملف JSON موحد
with open('data/fitness_knowledge_base.json', 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

print(f"Merged {len(all_data)} entries into data/fitness_knowledge_base.json") 