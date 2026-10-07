from oracle_experiment import load_data, score, compose
data = load_data()
dummy = [{"id": "x", "title": "none", "text": "(no inspiration given)"}]
for i in range(5):
    a, b = data[i], data[-(i + 1)]
    h = compose(a, dummy, 1)[0]
    print(a["id"], "| self:", score(a, a["gold_hypothesis"]),
          "| other paper:", score(a, b["gold_hypothesis"]),
          "| no inspiration:", score(a, h))