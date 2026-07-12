"""Gate an LLM/classifier on calibrated confidence -- answer or abstain."""

import random

from certain import ConformalClassifier, SelectiveAnswerer


def conformal_sets() -> None:
    print("== conformal prediction sets ==")
    rng = random.Random(0)
    classes = ["cat", "dog", "fox", "owl"]

    def synth(n):
        rows, ys = [], []
        for _ in range(n):
            y = rng.choice(classes)
            row = {c: rng.random() for c in classes}
            row[y] += 1.5 * rng.random()
            rows.append(row)
            ys.append(y)
        return rows, ys

    cal_rows, cal_ys = synth(1500)
    clf = ConformalClassifier(alpha=0.1)  # want 90% coverage
    clf.calibrate(cal_rows, cal_ys)

    for scores in (
        {"cat": 0.90, "dog": 0.05, "fox": 0.03, "owl": 0.02},   # confident
        {"cat": 0.35, "dog": 0.33, "fox": 0.30, "owl": 0.02},   # unsure
    ):
        print(f"  scores={scores}")
        print(f"    -> set {clf.predict_set(scores)}  (singleton = answer, "
              f"larger = flag for review)")


def selective_abstain() -> None:
    print("\n== selective answering ==")
    rng = random.Random(1)
    conf = [rng.random() for _ in range(2000)]
    correct = [rng.random() < 0.35 + 0.6 * c for c in conf]

    gate = SelectiveAnswerer(target_error=0.15)  # answer only if <=15% error
    gate.calibrate(conf, correct)
    print(f"  confidence threshold: {gate.threshold:.3f}")
    print(f"  answer at conf 0.95? {gate.answer(0.95)}")
    print(f"  answer at conf 0.40? {gate.answer(0.40)}")
    print(f"  {gate.evaluate(conf, correct)}")


if __name__ == "__main__":
    conformal_sets()
    selective_abstain()
