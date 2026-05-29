import bisect

class SortedList:
    def __init__(self):
        self._data = []

    def add(self, item):
        bisect.insort(self._data, item)

    def remove(self, item):
        i = bisect.bisect_left(self._data, item)
        if i != len(self._data) and self._data[i] == item:
            self._data.pop(i)
        else:
            raise ValueError(f"{item} not found")
        
    def search(self, item):
        i = bisect.bisect_left(self._data, item)
        return i < len(self._data) and self._data[i] == item
    
    def find_index(self, item):
        i = bisect.bisect_left(self._data, item)
        if i != len(self._data) and self._data[i] == item:
            return i
        return -1
    
    def range_search(self, low, high):
        left = bisect.bisect_left(self._data, low)
        right = bisect.bisect_right(self._data, high)
        return self._data[left:right]

    def __len__(self):
        return len(self._data)
    
    def __str__(self):
        return str(self._data)
    
    def __contains__(self, item):
        return self.search(item)
        

# Example usage
sl = SortedList()
for n in [5, 2, 8, 1, 9, 3, 7, 4, 6]:
    sl.add(n)

print(sl)                       # SortedList([1, 2, 3, 4, 5, 6, 7, 8, 9])
print(5 in sl)                  # True
print(10 in sl)                 # False
print(sl.find_index(7))         # 6
print(sl.range_search(3, 7))    # [3, 4, 5, 6, 7]
sl.remove(5)
print(sl)                       # SortedList([1, 2, 3, 4, 6, 7, 8, 9])