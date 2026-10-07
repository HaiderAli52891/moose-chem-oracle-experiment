from oracle_experiment_checklist import load_data, key_elements, judge_ask

data = load_data()
a = data[0]
els = key_elements(a)

print("KEY ELEMENTS FOUND:")
for i, e in enumerate(els, 1):
    print(i, e)

listing = "\n".join(f"{i + 1}. {e}" for i, e in enumerate(els))
out = judge_ask(f"""
Generated hypothesis:
{a['gold_hypothesis']}

Key elements of the ground-truth hypothesis:
{listing}

For each numbered element, answer Y only if the generated hypothesis uses the
SAME specific mechanism, material, or method. Answer N if it is only on the
same general topic or uses something different.
Reply in exactly this format: 1:Y 2:N 3:Y
""", 40)

print("\nRAW JUDGE ANSWER:", repr(out))