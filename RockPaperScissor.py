
def rpc(user_input): #RPC stands for Rock Paper Scissor
    import random
    dict1= {"Rock":1,"Paper":2,"Scissor":3}
    options = ["Rock", "Paper", "Scissor"]
    comp_response = random.choice(options)
    user=dict1[user_input]
    comp=dict1[comp_response]
    diff=user-comp

    if diff == 1 or diff == -2:
        result = "You Win!"

    elif diff == -1 or diff == 2:
        result = "you lose!"

    elif diff == 0:
        result = "Draw!"

    print("Computer:",comp_response,"|",result)
    return result

def program(win=0,lose=0,draw=0):
    user_entry = input(
        'Welcome to the game of Rock Paper Scissor!, \nPlease enter: "Rock", "Paper", or "Scissor"\nYour input:')
    validAnswer = ["Rock", "Paper", "Scissor"]
    a=win
    b=lose
    c=draw
    x=0
    y=0
    z=0
    while not user_entry in validAnswer:
        user_entry = input('Please Enter a Valid input from "Rock", "Paper", or "Scissor":')
        # validityTest = userInput in validAnswer
    else:
        tracker = rpc(user_entry)
        if tracker == "You Win!":
            a += 1
            x=1
        elif tracker == "you lose!":
            b += 1
            y=1
        elif tracker == "Draw!":
            c += 1
            z=1

        try_again = input('Would you like to try again? (Enter "Yes" or "No"):')
        validAnswer2 = ["Yes", "No"]
        while not try_again in validAnswer2:
            try_again = input('Please Specify "Yes" or "No":')
        else:
            if try_again == "Yes":
                output=program(a,b,c)
                win = output[0]
                lose = output[1]
                draw = output[2]


            elif try_again == "No":
                print("Thanks for playing, hope you had a good time!")
                win += x
                lose += y
                draw += z
    lst = [win, lose, draw]

    return lst


def scoreboard():
    test=program()
    print("Your Final Score: Win:",test[0],"| Lose:",test[1],"| Draw:",test[2])



scoreboard()










