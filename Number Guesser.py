import random
n = 10

Number = random.randint(0, n)
print("You have 3 guesses to guess a number from 0-"+str(n))
print()

guess_1 = int(input("Take a guess:"))
print()

if guess_1 == Number: 
    print("Correct, Welldone!")
elif guess_1 != Number:
    print("Wrong, Try again")
    print()
    print("Here have a clue:",end=" ")
    if Number % 2 == 0:
        print("Number is even")
    else:
        print("Number is odd")
    print()
    
    guess_2 = int(input("Take a guess:"))
    print()
    if guess_2 == Number:
        print("Correct, WELLDONE!")
    elif guess_2 != Number:
        print("Wrong again")
        print()
        print("Here have another clue:",end=' ')
        difference = abs(Number-guess_2)
        print("Your last guess was +/-"+str(difference)+" of the number")
        print()
   
        guess_3 = int(input("Enter your final guess:"))
        if guess_3 == Number:
                print()
                print("WELLDONE!")
        elif guess_3 != Number:
                print("Wrong :(, The number was "+str(Number))



