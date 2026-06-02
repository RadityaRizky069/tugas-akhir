import os
import numpy as np
import pickle
from modules.matching import match_features, make_decision, DECISION_THRESHOLD, MIN_INLIERS
from modules.feature_extraction import extract_sift_features
from modules.utils import deserialize_keypoints

DB_FILE = "palmprint_database.pkl"


def load_database():
    if not os.path.exists(DB_FILE):
        return {}
    try:
        with open(DB_FILE, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return {}


def _get_kp_desc(entry):
    kp_data = entry.get('keypoints')
    if kp_data is not None:
        return deserialize_keypoints(kp_data), entry['descriptors']
    roi = entry.get('roi_image')
    if roi is None or roi.size == 0:
        return None, None
    return extract_sift_features(roi)


def _compute_metrics(genuine_scores, impostor_scores):
    if not genuine_scores or not impostor_scores:
        return None

    tp = sum(1 for p, i in genuine_scores if p >= DECISION_THRESHOLD and i >= MIN_INLIERS)
    fn = len(genuine_scores) - tp
    fp = sum(1 for p, i in impostor_scores if p >= DECISION_THRESHOLD and i >= MIN_INLIERS)
    tn = len(impostor_scores) - fp

    total = tp + tn + fp + fn
    akurasi = (tp + tn) / total * 100 if total else 0.0
    presisi = tp / (tp + fp) * 100 if (tp + fp) else 100.0
    recall  = tp / (tp + fn) * 100 if (tp + fn) else 0.0
    far     = fp / (fp + tn) * 100 if (fp + tn) else 0.0

    return {
        'akurasi': akurasi,
        'presisi': presisi,
        'recall': recall,
        'far': far,
        'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
        'total_genuine': len(genuine_scores),
        'total_impostor': len(impostor_scores),
    }


def _display_table(m):
    print()
    print("=" * 65)
    print("          METRIK EVALUASI SISTEM VERIFIKASI PALMPRINT")
    print("=" * 65)
    print(f"  Threshold: >={DECISION_THRESHOLD:.0f}%  |  Min Inliers: >={MIN_INLIERS}")
    print("-" * 65)
    print(f"  {'Metrik Evaluasi':<30s} {'Hasil':>15s} {'Status':>15s}")
    print("-" * 65)

    def row(label, value, good, unit="%"):
        status = "[OK]" if good else "[NG]"
        print(f"  {label:<30s} {value:>13.2f}{unit}  {status:>15s}")

    row("Akurasi Sistem",         m['akurasi'], m['akurasi'] >= 85)
    row("Presisi (Genuine)",      m['presisi'], m['presisi'] >= 95)
    row("Recall (Genuine)",       m['recall'],  m['recall'] >= 80)
    row("False Acceptance (FAR)", m['far'],     m['far'] <= 1)

    print("-" * 65)
    print(f"  True Positive (TP) : {m['tp']:>3d}   | Total sampel genuine: {m['total_genuine']}")
    print(f"  False Negative (FN): {m['fn']:>3d}   | Total sampel impostor: {m['total_impostor']}")
    print(f"  False Positive (FP): {m['fp']:>3d}")
    print(f"  True Negative (TN) : {m['tn']:>3d}")
    print("=" * 65)
    print()


def main():
    db = load_database()
    users = list(db.keys())
    print(f"Loaded {len(users)} user(s): {users}")
    print()

    genuine_scores = []
    impostor_scores = []

    for user in users:
        entries = db[user]
        if not isinstance(entries, list):
            continue

        feats = []
        for e in entries:
            kp, desc = _get_kp_desc(e)
            if desc is not None and len(desc) > 0 and kp is not None and len(kp) > 0:
                feats.append((kp, desc))

        if len(feats) < 2:
            continue

        # Genuine: leave-one-out (best match per probe)
        for i in range(len(feats)):
            kp1, d1 = feats[i]
            best_inl = -1
            best_pct = 0.0
            for j in range(len(feats)):
                if i == j:
                    continue
                kp2, d2 = feats[j]
                inl, pct, _ = match_features(d1, d2, kp1, kp2)
                if inl > best_inl:
                    best_inl = inl
                    best_pct = pct
            if best_inl >= 0:
                genuine_scores.append((best_pct, best_inl))

        # Impostor: probe vs gallery of each other user
        for i in range(len(feats)):
            kp1, d1 = feats[i]
            for other_user in users:
                if other_user == user:
                    continue
                other_entries = db[other_user]
                if not isinstance(other_entries, list):
                    continue

                gallery = []
                for oe in other_entries:
                    kp2, d2 = _get_kp_desc(oe)
                    if d2 is not None and len(d2) > 0:
                        gallery.append((kp2, d2))
                if not gallery:
                    continue

                best_inl = -1
                best_pct = 0.0
                for kp2, d2 in gallery:
                    inl, pct, _ = match_features(d1, d2, kp1, kp2)
                    if inl > best_inl:
                        best_inl = inl
                        best_pct = pct
                if best_inl >= 0:
                    impostor_scores.append((best_pct, best_inl))

    print(f"Genuine scores  : {len(genuine_scores)}  (each probe vs best template of same user)")
    print(f"Impostor scores : {len(impostor_scores)}  (each probe vs best template of each other user)")
    print()

    if not genuine_scores or not impostor_scores:
        print("Not enough scores.")
        return

    m = _compute_metrics(genuine_scores, impostor_scores)
    if m is None:
        print("Could not compute metrics.")
        return

    _display_table(m)

    # EER search
    print("--- EER Search (varying DECISION_THRESHOLD) ---")
    min_diff = 1.0
    eer_th = None
    eer_far = eer_frr = None
    for th in range(0, 101):
        far_th = sum(1 for p, i in impostor_scores if p >= th and i >= MIN_INLIERS) / len(impostor_scores)
        frr_th = sum(1 for p, i in genuine_scores if p < th or i < MIN_INLIERS) / len(genuine_scores)
        diff = abs(far_th - frr_th)
        if diff < min_diff:
            min_diff = diff
            eer_th = th
            eer_far = far_th
            eer_frr = frr_th
    print(f"  EER at DECISION_THRESHOLD = {eer_th}% : FAR = {eer_far:.2%} , FRR = {eer_frr:.2%}")
    print()


if __name__ == "__main__":
    main()