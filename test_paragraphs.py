"""10-paragraph test of the letter-feature sentiment classifier.

Compares the letter-only random forest (trained on FPB binary) against
VADER (rule-based lexicon, tuned threshold) on 10 fresh, hand-written
financial paragraphs (5 positive, 5 negative, single-sentence FPB-style).

These are not in the FPB training set, so this is a held-out test of
the same distribution.
"""
from src.classify import classify_batch, vader_score

POSITIVE = [
    "The company reported record quarterly earnings, exceeding analyst expectations.",
    "The company announced strong revenue growth and improved profit margins for the year.",
    "The firm posted impressive financial results, beating all analyst forecasts.",
    "Profits surged to record levels driven by strong demand across all segments.",
    "Operating margin expanded significantly due to disciplined cost management.",
]

NEGATIVE = [
    "The company reported a major decline in quarterly earnings, missing analyst estimates.",
    "The firm announced significant layoffs and a substantial drop in revenue.",
    "The company posted disappointing financial results amid weakening market demand.",
    "Profits fell sharply as the company faced increased competition and rising costs.",
    "The firm recorded a large impairment charge and lowered its full-year guidance.",
]

texts = POSITIVE + NEGATIVE
true_labels = ["positive"] * 5 + ["negative"] * 5

# Run the letter classifier
results = classify_batch(texts)

# Run VADER for comparison
vader_results = [vader_score(t) for t in texts]

# Compare
print("=" * 100)
print("PARAGRAPH-LEVEL BINARY SENTIMENT CLASSIFIER: 10-PARAGRAPH TEST")
print("=" * 100)
print()
print(f"{'#':<3} {'TRUE':<10} {'LETTER':<14} {'P(pos)':<8} {'VADER':<14} {'COMP':<8}")
print("-" * 100)

n_correct_letter = 0
n_correct_vader = 0

for i, (text, true, res, vader) in enumerate(zip(texts, true_labels, results, vader_results), 1):
    pred = res["label"]
    p_pos = res.get("proba", {}).get("positive", 0.0)
    vader_pred = vader.get("label_tuned", "?")
    vader_comp = vader.get("compound", 0.0)
    match_l = pred == true
    match_v = vader_pred == true
    if match_l:
        n_correct_letter += 1
    if match_v:
        n_correct_vader += 1
    mark_l = "OK" if match_l else "X"
    mark_v = "OK" if match_v else "X"
    print(f"{i:<3} {true:<10} {pred:<8}{mark_l:<6} {p_pos:<8.3f} {vader_pred:<8}{mark_v:<6} {vader_comp:<+.3f}")
    snippet = text[:90] + "..." if len(text) > 90 else text
    print(f'     "{snippet}"')
    print()

print("-" * 100)
print(f"Letter classifier correct:  {n_correct_letter}/10")
print(f"VADER (tuned) correct:      {n_correct_vader}/10")
print()
print("Both agree with truth:      " + str(sum(1 for r, v, t in zip(results, vader_results, true_labels)
                                                  if r['label'] == t and v['label_tuned'] == t)) + "/10")
