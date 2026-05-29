class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
        self.size = 0

    def append(self, value):
        new_node = node(value)
        if self.head is None:
            self.head = new_node
            self.size += 1
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
        self.size += 1

    def prepend(self, value):
        new_node = node(value)
        new_node.next = self.head
        self.head = new_node
        self.size += 1

    def insert(self, index, value):
        if index < 0 or index > self.size:
            raise IndexError("Index out of bounds")
        new_node = node(value)
        if index == 0:
            self.prepend(value)
            return
        current = self.head
        for _ in range(index - 1):
            current = current.next
        new_node.next = current.next
        current.next = new_node
        self.size += 1

    def delete(self, value):
        if self.head is None:
            return False
        
        if self.head.value == value:
            self.head = self.head.next
            self.size -= 1
            return True

        current = self.head
        while current.next:
            if current.next.value == value:
                current.next = current.next.next
                self.size -= 1
                return True
            current = current.next
        raise ValueError("Value not found")

    def find(self, value):
        current = self.head
        index = 0
        while current:
            if current.value == value:
                return index
            current = current.next
            index += 1
        raise ValueError("Value not found")
    
    def reverse(self):
        prev = None
        current = self.head
        while current:
            next_node = current.next
            current.next = prev
            prev = current
            current = next_node
        self.head = prev

    def to_list(self):
        result = []
        current = self.head
        while current:
            result.append(current.value)
            current = current.next
        return result
    
    def __len__(self):
        return self.size
    
    def __str__(self):
        return " -> ".join(str(x) for x in self.to_list()) + " -> None"    
    def __contains__(self, value):
        current = self.head
        while current:
            if current.value == value:
                return True
            current = current.next
        return False
    

ll = LinkedList()
ll.append(1)
ll.append(2)
ll.append(3)
ll.append(4)
ll.prepend(0)

print(ll)               # 0 → 1 → 2 → 3 → 4 → None
print(len(ll))          # 5
print(ll.find(3))       # 3
print(3 in ll)          # True
print(99 in ll)         # False

ll.insert(2, 99)
print(ll)               # 0 → 1 → 99 → 2 → 3 → 4 → None

ll.delete(99)
print(ll)               # 0 → 1 → 2 → 3 → 4 → None

ll.reverse()
print(ll)               # 4 → 3 → 2 → 1 → 0 → None

print(ll.to_list())     # [4, 3, 2, 1, 0]