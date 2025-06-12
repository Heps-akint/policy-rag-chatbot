when assistant REPLY:
    # Must contain the tag...
    assert REPLY matches r"\[Sources\]\s*\n?"
    # ...followed by at least one bracketed citation e.g. [My_Policy.pdf]
    assert REPLY matches r"\[Sources\][\s\S]*\[[^\[\]]+\]"
