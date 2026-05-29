import random

number = random.randint(1, 9)
guess = 0
count = 0

while guess != number and guess!= "exit":
    guess =input("Guess a number between 1 and 9 (or type 'exit' to quit): ")
    if guess == "exit":
        print("Thanks for playing! Goodbye!")
        break
    guess =int(guess)
    count += 1
    if guess < number:
        print("Too low! Try again.")
    elif guess > number:
        print("Too high! Try again.")
    else:
        print(f"Congratulations! You've guessed the number {number} in {count} attempts!")