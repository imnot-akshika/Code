import random

a = [random.randint(1, 100) for _ in range(20)]

even = [x for x in a if x % 2 == 0]
even.sort()
print(even)