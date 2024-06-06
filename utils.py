import pickle
import faiss

def save_faiss_index(index, file_path):
    faiss.write_index(index, file_path)

def load_faiss_index(file_path):
    return faiss.read_index(file_path)

def save_message_storage(storage, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(storage, f)

def load_message_storage(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
