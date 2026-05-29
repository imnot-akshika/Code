def has_duplicate_fast(items):
    seen = set()
    for item in items:
        if item in seen:
            return True
        seen.add(item)
    return False

# Example 1
numbers = [1, 2, 3, 4, 2]

result = has_duplicate_fast(numbers)

print(result)
        



def get_common_fast(list1, list2):
    set2 = set(list2)
    return [a for a in list1 if a in set2]

# Example lists
numbers1 = [1, 2, 3, 4, 5]
numbers2 = [4, 5, 6, 7, 8]

# Run the function
result = get_common_fast(numbers1, numbers2)

# Print the result
print(result)