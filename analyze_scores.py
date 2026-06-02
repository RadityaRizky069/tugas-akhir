import os
import pickle
import cv2
import numpy as np
from modules.preprocessing import preprocess_and_extract_roi_detailed
from modules.feature_extraction import extract_sift_features
from modules.matching import match_features

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

def main():
    db = load_database()
    genuine = []
    impostor = []
    
    for user, entries in db.items():
        if not isinstance(entries, list):
            continue
        # Build list of features for this user
        feats = []
        for entry in entries:
            roi, kp, desc = get_roi_and_features_from_entry(entry)
            if roi is None:
                continue
            feats.append((roi, kp, desc))
        if len(feats) < 2:
            continue
        # Genuine: leave-one-out
        for i in range(len(feats)):
            probe_roi, probe_kp, probe_desc = feats[i]
            best_inliers = -1
            best_pct = 0.0
            for j in range(len(feats)):
                if i == j:
                    continue
                roi, kp, desc = feats[j]
                inliers, pct, _ = match_features(probe_desc, desc, probe_kp, kp)
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_pct = pct
            if best_inliers >= 0:
                genuine.append((best_pct, best_inliers))
        # Impostor: probe from this user vs each other user
        for i in range(len(feats)):
            probe_roi, probe_kp, probe_desc = feats[i]
            for other_user, other_entries in db.items():
                if other_user == user:
                    continue
                if not isinstance(other_entries, list):
                    continue
                # build gallery for other_user
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
                    inliers, pct, _ = match_features(probe_desc, desc, probe_kp, kp)
                    if inliers > best_inliers:
                        best_inliers = inliers
                        best_pct = pct
                if best_inliers >= 0:
                    impostor.append((best_pct, best_inliers))
    
    print(f"Genuine count: {len(genuine)}")
    print(f"Impostor count: {len(impostor)}")
    if genuine:
        genuine_pct = [p for p,_ in genuine]
        genuine_inp = [i for _,i in genuine]
        print(f"Genuine percentage: min={min(genuine_pct):.1f}, max={max(genuine_pct):.1f}, mean={np.mean(genuine_pct):.1f}")
        print(f"Genuine inliers: min={min(genuine_inp)}, max={max(genuine_inp)}, mean={np.mean(genuine_inp):.1f}")
    if impostor:
        impostor_pct = [p for p,_ in impostor]
        impostor_inp = [i for _,i in impostor]
        print(f"Impostor percentage: min={min(impostor_pct):.1f}, max={max(impostor_pct):.1f}, mean={np.mean(impostor_pct):.1f}")
        print(f"Impostor inliers: min={min(impostor_inp)}, max={max(impostor_inp)}, mean={np.mean(impostor_inp):.1f}")

if __name__ == "__main__":
    main()