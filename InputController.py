name = input("What is your name? ")
print("Hello, " + name + "! Welcome to the game.")

age = int(input("How old are you? "))
year = int(2026 - age + 100)

print(f"You will turn 100 years old in the year {year}.\n" * age)