import os
import cv2
import numpy as np
import pickle
from modules.preprocessing import preprocess_and_extract_roi_detailed
from modules.feature_extraction import extract_sift_features
from modules.matching import match_features, make_decision

DB_FILE = "palmprint_database.pkl"

def load_database():
    if not os.path.exists(DB_FILE):
        return {}
    try:
        with open(DB_FILE, 'rb') as f:
            return pickle.load(f)
    except:
        return {}

def get_roi_and_features(image_path):
    """Return (roi, kp, desc) for given image path."""
    roi, steps = preprocess_and_extract_roi_detailed(image_path)
    if roi is None:
        return None, None, None
    kp, desc = extract_sift_features(roi)
    return roi, kp, desc

def verify_user(probe_image_path, claimed_user, db):
    """Return (decision, percentage, inliers) for probe vs claimed user's templates."""
    probe_roi, probe_kp, probe_desc = get_roi_and_features(probe_image_path)
    if probe_roi is None:
        return False, 0.0, 0  # failed to process
    
    if claimed_user not in db:
        return False, 0.0, 0
    
    entries = db[claimed_user]
    if not isinstance(entries, list) or len(entries) == 0:
        return False, 0.0, 0
    
    best_inliers = -1
    best_pct = 0.0
    
    for entry in entries:
        template_roi = entry['roi_image']
        if template_roi is None or template_roi.size == 0:
            continue
        template_kp, template_desc = extract_sift_features(template_roi)
        if template_desc is None or len(template_kp) == 0:
            continue
        
        inliers, pct, _ = match_features(probe_desc, template_desc, probe_kp, template_kp)
        if inliers > best_inliers:
            best_inliers = inliers
            best_pct = pct
    
    if best_inliers < 0:
        return False, 0.0, 0
    
    decision = make_decision(best_pct, best_inliers)
    return decision, best_pct, best_inliers

def main():
    db = load_database()
    print(f"Loaded {len(db)} users: {list(db.keys())}")
    
    genuine_scores = []  # list of (percentage, inliers)
    impostor_scores = []
    
    # For each user, treat each image as probe
    for user, entries in db.items():
        if not isinstance(entries, list):
            continue
        # We need the original image paths to re-extract ROI; but database only stores ROI and desc.
        # We cannot get the original path from the stored entry.
        # So we cannot do leave-one-out without the original images.
        # Instead, we will use the stored ROI as probe? That would be cheating because we are using the same ROI that was used to create the template.
        # However, the verification pipeline in the GUI uses the probe image (which could be the same as a template image) and compares with templates.
        # If we use the same ROI as probe and it matches itself, we expect a high score.
        # For genuine, we can use each stored ROI as probe and compare with the other stored ROIs of the same user.
        # For impostor, we use each stored ROI as probe and compare with the templates of other users.
        
        # Since we don't have the original image paths, we will use the stored ROI as the probe image.
        # This is acceptable for evaluating the matching logic because the preprocessing is deterministic.
        
        # Build a list of (roi, kp, desc) for this user's entries
        user_rois = []
        for entry in entries:
            roi = entry['roi_image']
            if roi is None or roi.size == 0:
                continue
            kp, desc = extract_sift_features(roi)
            if desc is None or len(kp) == 0:
                continue
            user_rois.append((roi, kp, desc))
        
        if len(user_rois) < 2:
            # Need at least 2 to do leave-one-out
            continue
        
        # Genuine: leave-one-out within same user
        for i in range(len(user_rois)):
            probe_roi, probe_kp, probe_desc = user_rois[i]
            # Gallery is all other entries of the same user
            best_inliers = -1
            best_pct = 0.0
            for j in range(len(user_rois)):
                if i == j:
                    continue
                roi, kp, desc = user_rois[j]
                inliers, pct, _ = match_features(probe_desc, desc, probe_kp, kp)
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_pct = pct
            if best_inliers >= 0:
                genuine_scores.append((best_pct, best_inliers))
        
        # Impostor: probe from this user, gallery from each other user
        for i in range(len(user_rois)):
            probe_roi, probe_kp, probe_desc = user_rois[i]
            for other_user, other_entries in db.items():
                if other_user == user:
                    continue
                if not isinstance(other_entries, list):
                    continue
                # Build gallery for other_user
                gallery = []
                for entry in other_entries:
                    roi = entry['roi_image']
                    if roi is None or roi.size == 0:
                        continue
                    kp, desc = extract_sift_features(roi)
                    if desc is None or len(kp) == 0:
                        continue
                    gallery.append((kp, desc))
                if not gallery:
                    continue
                # Best match against this other user's gallery
                best_inliers = -1
                best_pct = 0.0
                for kp, desc in gallery:
                    inliers, pct, _ = match_features(probe_desc, desc, probe_kp, kp)
                    if inliers > best_inliers:
                        best_inliers = inliers
                        best_pct = pct
                if best_inliers >= 0:
                    impostor_scores.append((best_pct, best_inliers))
    
    print(f"Genuine scores: {len(genuine_scores)}")
    print(f"Impostor scores: {len(impostor_scores)}")
    
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        print("Not enough scores to compute metrics.")
        return
    
    # Compute FAR and FRR at thresholds from 0 to 100 for percentage and inliers
    # We'll use the same thresholds as in make_decision: percentage >= DECISION_THRESHOLD and inliers >= MIN_INLIERS
    from modules.matching import DECISION_THRESHOLD, MIN_INLIERS
    print(f"\nUsing threshold: percentage >= {DECISION_THRESHOLD}, inliers >= {MIN_INLIERS}")
    
    # Genuine acceptance rate (GAR) = proportion of genuine that pass
    genuine_pass = sum(1 for pct, inp in genuine_scores if pct >= DECISION_THRESHOLD and inp >= MIN_INLIERS)
    gar = genuine_pass / len(genuine_scores) if genuine_scores else 0
    # False rejection rate (FRR) = 1 - GAR
    frr = 1 - gar
    
    # False acceptance rate (FAR) = proportion of impostor that pass
    impostor_pass = sum(1 for pct, inp in impostor_scores if pct >= DECISION_THRESHOLD and inp >= MIN_INLIERS)
    far = impostor_pass / len(impostor_scores) if impostor_scores else 0
    
    print(f"Genuine Pass: {genuine_pass}/{len(genuine_scores)} -> GAR: {gar:.2%}, FRR: {frr:.2%}")
    print(f"Impostor Pass: {impostor_pass}/{len(impostor_scores)} -> FAR: {far:.2%}")
    
    # Also compute EER by searching over thresholds (simplistic)
    # We'll vary percentage threshold from 0 to 100 and keep MIN_INLIERS fixed.
    # For each percentage threshold, compute FAR and FRR (based on both conditions).
    # Find where FAR and FRR are closest.
    print("\n--- Searching for EER (varying percentage threshold, MIN_INLIERS fixed) ---")
    min_diff = 1.0
    eer_thresh = None
    eer_far = eer_frr = None
    for thresh in range(0, 101):
        far_thresh = sum(1 for pct, inp in impostor_scores if pct >= thresh and inp >= MIN_INLIERS) / len(impostor_scores)
        frr_thresh = sum(1 for pct, inp in genuine_scores if pct < thresh or inp < MIN_INLIERS) / len(genuine_scores)
        diff = abs(far_thresh - frr_thresh)
        if diff < min_diff:
            min_diff = diff
            eer_thresh = thresh
            eer_far = far_thresh
            eer_frr = frr_thresh
    print(f"EER at percentage threshold {eer_thresh}: FAR={eer_far:.2%}, FRR={eer_frr:.2%}")

if __name__ == "__main__":
    main()