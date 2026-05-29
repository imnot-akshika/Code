print("This program will determine if a number is odd or even.")
print("Welsome to Odd or Even!")

number = int(input("Please enter a number: "))
if number % 2 == 0:
    print("The number you entered is even.")
else:
    print("The number you entered is odd.")

if number % 4 == 0:
    print("The number you entered is also divisible by 4.")
else:
    print("Not friendly with 4.")


num1 = int(input("Please enter another number: "))
num2 = int(input("Please enter one more number: "))
div = num1 / num2
if div % 2 == 0:
    print("The result of the division is even." + " The result is: " + str(div))
else:
    print("The result of the division is odd." + " The result is: " + str(div))
