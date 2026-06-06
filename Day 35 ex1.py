import random
import time


def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i -1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

def selection_sort(arr):
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr

def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

def merge_sort(arr):
    if len(arr) <=1:
        return arr
    
    mid = len(arr) // 2
    left_half = merge_sort(arr[:mid])
    right_half = merge_sort(arr[mid:])
    return merge(left_half, right_half)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quick_sort(left) + middle + quick_sort(right)

def test_sort(sort_fn, name):
    # test 1 — random list
    arr = random.sample(range(100), 10)
    assert sort_fn(arr.copy()) == sorted(arr), f"{name} failed random test"

    # test 2 — already sorted
    arr = list(range(10))
    assert sort_fn(arr.copy()) == arr, f"{name} failed sorted test"

    # test 3 — reverse sorted
    arr = list(range(10, 0, -1))
    assert sort_fn(arr.copy()) == sorted(arr), f"{name} failed reverse test"

    # test 4 — duplicates
    arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
    assert sort_fn(arr.copy()) == sorted(arr), f"{name} failed duplicates test"

    print(f"{name}: all tests passed")

test_sort(bubble_sort, "Bubble Sort")
test_sort(selection_sort, "Selection Sort")
test_sort(insertion_sort, "Insertion Sort")
test_sort(merge_sort, "Merge Sort")
test_sort(quick_sort, "Quick Sort")

def benchmark_sorts():
    arr = random.sample(range(10000), 1000)
    
    sorts = [
        (bubble_sort, "Bubble Sort"),
        (selection_sort, "Selection Sort"),
        (insertion_sort, "Insertion Sort"),
        (merge_sort, "Merge Sort"),
        (quick_sort, "Quicksort"),
    ]
    
    for sort_fn, name in sorts:
        test_arr = arr.copy()
        start = time.perf_counter()
        sort_fn(test_arr)
        elapsed = time.perf_counter() - start
        print(f"{name}: {elapsed:.4f}s")

benchmark_sorts()