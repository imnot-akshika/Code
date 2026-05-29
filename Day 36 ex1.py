def linear_search(arr, target):
    for i, item in enumerate(arr):
        if item == target:
            return i
    return -1
def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1

def recursive_binary_search(arr, target, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    if low > high:
        return -1
    
    mid = (low + high) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return recursive_binary_search(arr, target, mid + 1, high)
    else:
        return recursive_binary_search(arr, target, low, mid - 1)
    
import random
arr = sorted(random.sample(range(1000), 20))
target = random.choice(arr)
missing = 999

print(f"Array: {arr}")
print(f"Linear search {target}: {linear_search(arr, target)}")
print(f"Binary search {target}: {binary_search(arr, target)}")
print(f"Recursive binary {target}: {recursive_binary_search(arr, target)}")
print(f"Linear search {missing}: {linear_search(arr, missing)}")
print(f"Binary search {missing}: {binary_search(arr, missing)}")