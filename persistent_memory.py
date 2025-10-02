# persistent_memory.py
import json, os

class PersistentMemory:
    def __init__(self, file_path="history/chat_history.json"):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.file_path = file_path
        self.history = self.load_history()

    def load_history(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def save_history(self):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)

    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})
        self.save_history()

    def get_messages(self):
        return self.history

    def clear(self):
        self.history = []
        self.save_history()
