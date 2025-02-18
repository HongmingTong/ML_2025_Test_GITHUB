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

def program():
    user_entry = input(
        'Welcome to the game of Rock Paper Scissor!, \nPlease enter: "Rock", "Paper", or "Scissor"\nYour input:')
    validAnswer = ["Rock", "Paper", "Scissor"]

    while not user_entry in validAnswer:
        user_entry = input('Please Enter a Valid input from "Rock", "Paper", or "Scissor":')
        # validityTest = userInput in validAnswer
    else:
        rpc(user_entry)

        try_again = input('Would you like to try again? (Enter "Yes" or "No"):')
        validAnswer2 = ["Yes", "No"]
        while not try_again in validAnswer2:
            try_again = input('Please Specify "Yes" or "No":')
        else:
            if try_again == "Yes":
                program()
            elif try_again == "No":
                print("Thanks for playing, hope you had a good time!")


    return


program()










