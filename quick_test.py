import os
import pickle
import cv2
import numpy as np
from modules.preprocessing import preprocess_and_extract_roi_detailed
from modules.feature_extraction import extract_sift_features

DB_FILE = "palmprint_database.pkl"

def load_database():
    if not os.path.exists(DB_FILE):
        return {}
    try:
        with open(DB_FILE, 'rb') as f:
            return pickle.load(f)
    except:
        return {}

def get_roi_and_features_from_entry(entry):
    roi = entry['roi_image']
    if roi is None or roi.size == 0:
        return None, None, None
    kp, desc = extract_sift_features(roi)
    return roi, kp, desc

def evaluate(decision_thresh, min_inliers, lowe_ratio):
    db = load_database()
    genuine = []
    impostor = []
    
    for user, entries in db.items():
        if not isinstance(entries, list):
            continue
        feats = []
        for entry in entries:
            roi, kp, desc = get_roi_and_features_from_entry(entry)
            if roi is None:
                continue
            feats.append((roi, kp, desc))
        if len(feats) < 2:
            continue
        # genuine leave-one-out
        for i in range(len(feats)):
            probe_roi, probe_kp, probe_desc = feats[i]
            best_inliers = -1
            best_pct = 0.0
            for j in range(len(feats)):
                if i == j:
                    continue
                roi, kp, desc = feats[j]
                # compute matches with current params
                bf = cv2.BFMatcher(cv2.NORM_L2)
                raw = bf.knnMatch(probe_desc, desc, k=2)
                good = []
                for pair in raw:
                    if len(pair)==2:
                        m,n = pair
                        if m.distance < lowe_ratio * n.distance:
                            good.append(m)
                if len(good) < 4:
                    continue
                src = np.float32([probe_kp[m.queryIdx].pt for m in good]).reshape(-1,1,2)
                dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1,1,2)
                H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                if H is None or mask is None:
                    continue
                inliers = int(mask.sum())
                pct = min(100.0, (inliers / len(good)) * 100.0)
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_pct = pct
            if best_inliers >= 0:
                genuine.append((best_pct, best_inliers))
        # impostor
        for i in range(len(feats)):
            probe_roi, probe_kp, probe_desc = feats[i]
            for other_user, other_entries in db.items():
                if other_user == user:
                    continue
                if not isinstance(other_entries, list):
                    continue
                gallery = []
                for entry in other_entries:
                    roi, kp, desc = get_roi_and_features_from_entry(entry)
                    if roi is None:
                        continue
                    gallery.append((kp, desc))
                if not gallery:
                    continue
                best_inliers = -1
                best_pct = 0.0
                for kp, desc in gallery:
                    bf = cv2.BFMatcher(cv2.NORM_L2)
                    raw = bf.knnMatch(probe_desc, desc, k=2)
                    good = []
                    for pair in raw:
                        if len(pair)==2:
                            m,n = pair
                            if m.distance < lowe_ratio * n.distance:
                                good.append(m)
                    if len(good) < 4:
                        continue
                    src = np.float32([probe_kp[m.queryIdx].pt for m in good]).reshape(-1,1,2)
                    dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1,1,2)
                    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                    if H is None or mask is None:
                        continue
                    inliers = int(mask.sum())
                    pct = min(100.0, (inliers / len(good)) * 100.0)
                    if inliers > best_inliers:
                        best_inliers = inliers
                        best_pct = pct
                if best_inliers >= 0:
                    impostor.append((best_pct, best_inliers))
    
    if not genuine or not impostor:
        return None
    genuine_pass = sum(1 for pct,inp in genuine if pct >= decision_thresh and inp >= min_inliers)
    gar = genuine_pass / len(genuine)
    frr = 1 - gar
    impostor_pass = sum(1 for pct,inp in impostor if pct >= decision_thresh and inp >= min_inliers)
    far = impostor_pass / len(impostor)
    return frr, far, gar, len(genuine), len(impostor)

def main():
    # Try some combos
    combos = [
        (70, 5, 0.75),
        (70, 5, 0.6),
        (70, 8, 0.6),
        (80, 5, 0.6),
        (80, 8, 0.6),
        (75, 6, 0.6),
        (70, 4, 0.6),
        (60, 4, 0.6),
    ]
    for thresh, minin, lowe in combos:
        res = evaluate(thresh, minin, lowe)
        if res is None:
            print(f"th={thresh}, minIn={minin}, lowe={lowe} -> insufficient data")
            continue
        frr, far, gar, n_gen, n_imp = res
        print(f"th={thresh}, minIn={minin}, lowe={lowe:.2f} -> FRR={frr:.2%}, FAR={far:.2%}, GAR={gar:.2%} (gen={n_gen}, imp={n_imp})")

if __name__ == "__main__":
    main()