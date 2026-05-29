class DynamicArray:
    def __init__(self):
        self._capacity = 2      # start with space for 2 items
        self._size = 0
        self._data = [None] * self._capacity

    def append(self, item):
        if self._size == self._capacity:
            self._resize()
        self._data[self._size] = item
        self._size += 1

    def _resize(self):
        # double the capacity
        self._capacity *= 2
        new_data = [None] * self._capacity
        for i in range(self._size):
            new_data[i] = self._data[i]
        self._data = new_data
        print(f"  [resized to capacity {self._capacity}]")

    def get(self, index):
        if index < 0 or index >= self._size:
            raise IndexError(f"Index {index} out of range.")
        return self._data[index]

    def __len__(self):
        return self._size

    def __str__(self):
        return str(self._data[:self._size])
    
    def insert(self, index, item):
        if index < 0 or index > self._size:
            raise IndexError(f"Index {index} out of range.")
        if self._size == self._capacity:
            self._resize()
        # Shift items to the right
        for i in range(self._size, index, -1):
            self._data[i] = self._data[i - 1]
        self._data[index] = item
        self._size += 1

    def delete(self, index):
        if index < 0 or index >= self._size:
            raise IndexError(f"Index {index} out of range.")
        # Shift items to the left
        for i in range(index, self._size - 1):
            self._data[i] = self._data[i + 1]
        self._data[self._size - 1] = None  # Clear last item
        self._size -= 1

    def contains(self, item):
        for i in range(self._size):
            if self._data[i] == item:
                return True
        return False
    

arr = DynamicArray()
arr.append(1)
arr.append(2)
arr.append(3)
arr.append(4)
arr.append(5)

print(arr)              # [1, 2, 3, 4, 5]
print(len(arr))         # 5
print(arr.get(2))       # 3
print(arr.contains(4))  # True

arr.insert(2, 99)
print(arr)              # [1, 2, 99, 3, 4, 5]

arr.delete(0)
print(arr)              # [2, 99, 3, 4, 5]