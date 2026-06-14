def digital_root(n):
    if n > 9:
        return digital_root(sum(int(digits) for digits in str(n)))
    for digits in str(n):
        return sum(int(digits) for digits in str(n))
    
print(digital_root(16))  # Output: 7
print(digital_root(942)) # Output: 6