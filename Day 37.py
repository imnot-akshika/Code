from functools import lru_cache

@lru_cache(maxsize=None)
def climbing_stairs(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    elif n == 2:
        return 2
    else:
        return climbing_stairs(n - 1) + climbing_stairs(n - 2)
    

print(climbing_stairs(1))   # 1
print(climbing_stairs(5))   # 8
print(climbing_stairs(10))  # 89
print(climbing_stairs(20))  # 10946

def house_robber(houses):
    @lru_cache(maxsize=None)
    def dp(i):
        if i >= len(houses):
            return 0
        return max(
            houses[i] + dp(i + 2),  # Rob current house and skip the next one
            dp(i + 1)               # Skip the current house
        )
    return dp(0)

print(house_robber([2, 7, 9, 3, 1]))
print(house_robber([1, 2, 3, 1]))

def word_break(s, words):
    word_set = set(words)
    @lru_cache(maxsize=None)
    def dp(start):
        if start == len(s):
            return True
        for end in range(start + 1, len(s) + 1):
            if s[start:end] in word_set and dp(end):
                return True
        return False
    return dp(0)

print(word_break("leetcode", {"leet", "code"}))  # True
print(word_break("applepenapple", {"apple", "pen"}))  # True
print(word_break("catsandog", {"cats", "dog", "sand", "and", "cat"}))  # False