import os
import json
import argparse
from typing import List, Dict


class ChatHistory:
    BASE_DIR = "data/chat_sessions"

    def __init__(self, session_id: str):
        self.session_id = session_id
        os.makedirs(self.BASE_DIR, exist_ok=True)
        self.file_path = os.path.join(self.BASE_DIR, f"{session_id}.json")

    # -----------------------------------------------------------
    # Create file if not exists
    # -----------------------------------------------------------
    def create_if_not_exists(self):
        """Create empty history file if missing."""
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)

    # -----------------------------------------------------------
    # Load History
    # -----------------------------------------------------------
    def get_history(self) -> List[Dict]:
        if not os.path.exists(self.file_path):
            return []
        with open(self.file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # -----------------------------------------------------------
    # Add Message
    # -----------------------------------------------------------
    def add_message(self, role: str, content: str):
        history = self.get_history()
        history.append({"role": role, "content": content})
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    # -----------------------------------------------------------
    # Clear Chat
    # -----------------------------------------------------------
    def clear(self):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)

    # -----------------------------------------------------------
    # Delete Session File
    # -----------------------------------------------------------
    @staticmethod
    def delete_session(session_id: str):
        file_path = os.path.join(ChatHistory.BASE_DIR, f"{session_id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)

    # -----------------------------------------------------------
    # List all sessions
    # -----------------------------------------------------------
    @staticmethod
    def list_sessions() -> List[str]:
        if not os.path.exists(ChatHistory.BASE_DIR):
            return []
        files = os.listdir(ChatHistory.BASE_DIR)
        return sorted(f.replace(".json", "") for f in files if f.endswith(".json"))

    # -----------------------------------------------------------
    # EXPORT CHAT → TXT FILE
    # -----------------------------------------------------------
    def export(self, export_path: str = None):
        """Export chat history to a .txt file with clean formatting."""

        history = self.get_history()
        if not history:
            raise ValueError(f"No messages found for session: {self.session_id}")

        if export_path is None:
            export_path = f"{self.session_id}_export.txt"

        lines = []
        lines.append(f"CHAT SESSION EXPORT: {self.session_id}")
        lines.append("=" * 60)
        lines.append("")

        for msg in history:
            role = msg["role"].upper()
            content = msg["content"]
            lines.append(f"{role}:")
            lines.append(content)
            lines.append("")  # blank line between messages

        with open(export_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return export_path


# -----------------------------------------------------------
# CLI SUPPORT
# -----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat Session Export Tool")
    parser.add_argument("--export", type=str, help="Export chat session to text file")

    args = parser.parse_args()

    if args.export:
        session_id = args.export
        chat = ChatHistory(session_id)

        try:
            out_file = chat.export()
            print(f"✅ Exported chat session '{session_id}' → {out_file}")
        except Exception as e:
            print(f"❌ Error: {e}")
