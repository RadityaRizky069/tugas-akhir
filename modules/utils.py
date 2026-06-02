import pickle
import os
import cv2
import numpy as np

DB_FILE = "palmprint_database.pkl"


def load_database():
    if not os.path.exists(DB_FILE):
        return {}
    try:
        with open(DB_FILE, 'rb') as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError, ModuleNotFoundError):
        return {}


def save_database(data):
    with open(DB_FILE, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def register_user(username, entries):
    data = load_database()
    data[username] = entries
    save_database(data)
    return True


def get_user_data(username):
    data = load_database()
    return data.get(username, [])


def get_all_users():
    data = load_database()
    return list(data.keys())


def delete_user(username):
    data = load_database()
    if username in data:
        del data[username]
        save_database(data)
        return True
    return False


def serialize_keypoints(keypoints):
    return [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id)
            for kp in keypoints]


def deserialize_keypoints(data):
    if data is None:
        return []
    return [cv2.KeyPoint(x, y, size, angle, response, octave, class_id)
            for x, y, size, angle, response, octave, class_id in data]
