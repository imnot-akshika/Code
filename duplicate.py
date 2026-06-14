def duplicate_count(text):
    t = text.lower()
    count = 0

    for char in set(t):
        if t.count(char) > 1:
            count += 1
    return count
