def sum_two_smallest_numbers(numbers):
    min1, min2 = sorted(numbers)[:2]
    return min1 + min2

print(sum_two_smallest_numbers([10, 343445353, 3453445, 3453545353453]))