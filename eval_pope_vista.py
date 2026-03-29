import json, glob
files = glob.glob('experiments/pope_full_9000/vista_method_9000/raw_*.jsonl')
overall_correct = 0
overall_total = 0
for f in files:
    with open(f) as fp:
        lines_data = [json.loads(line) for line in fp]
    correct = 0
    total = 0
    for d in lines_data:
        ans = d.get('ans', d.get('text', '')).strip().lower()
        label = d['label'] # 1=yes, 0=no
        # sometimes answer is "yes, ..." or "no, ..."
        pred = 1 if ans.startswith('yes') else 0 if ans.startswith('no') else -1
        if pred == label:
            correct += 1
        total += 1
    overall_correct += correct
    overall_total += total
    print(f"{f}: {correct/total:.4f}")
print(f"Overall VISTA: {overall_correct/overall_total:.4f}")
