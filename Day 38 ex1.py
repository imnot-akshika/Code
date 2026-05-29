def read_numbers(limit):
    for i in range(1, limit + 1):
        yield i


def filter_even(numbers):
    for number in numbers:
        if number % 2 == 0:
            yield number


def square(numbers):
    for number in numbers:
        yield number ** 2


pipeline = square(filter_even(read_numbers(20)))
print(list(pipeline))