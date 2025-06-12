when user MESSAGE:
    if MESSAGE contains_any ["medical advice", "legal advice",
                             "medical", "legal", "medicine", "lawyer"]:
        assistant REPLY "Sorry, I can’t provide that."