import random

number1 = int(input("Enter the number of elements in list A: "))
number2 = int(input("Enter the number of elements in list B: "))

a = [random.randint(1, 50) for _ in range(number1)]
print(a)
b = [random.randint(1, 50) for _ in range(number2)]
print(b)


for x in set(a):
    if x in set(b):
        print(x)

