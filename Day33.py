class HashMap:
    def __init__(self, capacity=8):
        self._capacity = capacity
        self._size = 0
        self._buckets = [None] * capacity    # array of buckets

    def _hash(self, key):
        return hash(key) % self._capacity    # map key to bucket index

    def set(self, key, value):
        index = self._hash(key)

        # linear probing — find empty slot or existing key
        while self._buckets[index] is not None:
            if self._buckets[index][0] == key:
                self._buckets[index] = (key, value)  # update existing
                return
            index = (index + 1) % self._capacity     # next bucket

        self._buckets[index] = (key, value)
        self._size += 1

        # resize if load factor > 0.66
        if self._size / self._capacity > 0.66:
            self._resize()

    def get(self, key, default=None):
        index = self._hash(key)

        while self._buckets[index] is not None:
            if self._buckets[index][0] == key:
                return self._buckets[index][1]
            index = (index + 1) % self._capacity

        return default    # key not found

    def _resize(self):
        old_buckets = self._buckets
        self._capacity *= 2
        self._buckets = [None] * self._capacity
        self._size = 0

        for bucket in old_buckets:
            if bucket is not None:
                self.set(bucket[0], bucket[1])    # re-hash everything

    def __setitem__(self, key, value):
        self.set(key, value)

    def __getitem__(self, key):
        result = self.get(key)
        if result is None:
            raise KeyError(key)
        return result

    def __len__(self):
        return self._size

    def __str__(self):
        items = [b for b in self._buckets if b is not None]
        return "{" + ", ".join(f"{k}: {v}" for k, v in items) + "}"
    
    def delete(self,key):
        index = self._hash(key)

        while self._buckets[index] is not None:
            if self._buckets[index][0] == key:
                self._buckets[index] = None  # mark as deleted
                self._size -= 1
                return
            index = (index + 1) % self._capacity
        raise KeyError(key)  # key not found

    def contains(self, key):
        result = self.get(key)
        if result is None:
            return False
        return True
    
    def keys(self):
        return [b[0] for b in self._buckets if b is not None]
    def values(self):
        return [b[1] for b in self._buckets if b is not None]
    def items(self):
        return [b for b in self._buckets if b is not None]
    def __contains__(self, item):
        return self.contains(item)


class WordFrequency:
    def __init__(self):
        self._freq_map = HashMap()

    def add_text(self, text):
        words = text.split()
        for word in words:
            word = word.lower().strip(".,!?;:")
            current_freq = self._freq_map.get(word, 0)
            self._freq_map[word] = current_freq + 1
        

    def top_n(self, n):
        items = self._freq_map.items()
        items.sort(key=lambda x: x[1], reverse=True)
        return items[:n]
    
    def frequency(self, word):
        return self._freq_map.get(word.lower(), 0)
    
    def unique_words(self):
        return len(self._freq_map)
    
    def __str__(self):
        total = sum(self._freq_map.values())
        return f"WordFrequency | {total} words | {len(self._freq_map)} unique"



# example usage
wf = WordFrequency()
wf.add_text("the quick brown fox jumps over the lazy dog the fox")
wf.add_text("the dog barked at the fox")

print(wf.frequency("the"))      # 6
print(wf.frequency("fox"))      # 3
print(wf.unique_words())        # 8
print(wf.top_n(3))              # [('the', 6), ('fox', 3), ('dog', 2)]
print(wf)                       # WordFrequency | 15 words | 8 unique