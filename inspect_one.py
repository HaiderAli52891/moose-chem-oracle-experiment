from oracle_experiment import load_data, compose, score
data = load_data()
a = data[0]
dummy = [{"id": "x", "title": "none", "text": "(no inspiration given)"}]
h = compose(a, dummy, 1)[0]
print("GOLD HYPOTHESIS:\n", a["gold_hypothesis"][:1500])
print("\nGENERATED WITHOUT ANY INSPIRATION:\n", h[:1500])
print("\nJUDGE SCORE:", score(a, h))