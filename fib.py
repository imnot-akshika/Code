def fibonacci_iterative(n_terms):
    a, b = 0, 1
    count = 0
    if n_terms <= 0:
        print("Please enter a positive integer")
    elif n_terms == 1:
        print(a)
    else:
        while count < n_terms:
            print(a, end=" ")
            # Update values: next number is the sum of previous two
            a, b = b, a + b
            count += 1



num = int(input("Enter the number of terms: "))
fibonacci_iterative(num)