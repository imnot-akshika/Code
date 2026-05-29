# write a generator that yields fibonacci numbers forever
def fibonacci_gen():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

# take first 10
gen = fibonacci_gen()
first_ten = [next(gen) for _ in range(10)]
print(first_ten)    # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

# find first fibonacci number over 1000
gen = fibonacci_gen()
result = next(n for n in gen if n > 1000)
print(result)    # 1597