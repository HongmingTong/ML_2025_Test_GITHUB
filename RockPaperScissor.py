def rpc(user_input): #RPC stands for Rock Paper Scissor | the function input is the users input
    import random
    dict1= {"Rock":1,"Paper":2,"Scissor":3}  # Create a dictionary to digitize users and computers response
    options = ["Rock", "Paper", "Scissor"]   # List of options for Computer's response
    comp_response = random.choice(options)   # Select Computer's response using random function
    user=dict1[user_input]    #Digitze User's input
    comp=dict1[comp_response] #Digitize Computers response
    diff=user-comp            # Compute the numerical difference between user and computer

    if diff == 1 or diff == -2:  # determine win lose draw based on the numerical difference
        result = "You Win!"
    elif diff == -1 or diff == 2:
        result = "you lose!"
    elif diff == 0:
        result = "Draw!"

    print("Computer:",comp_response,"|",result)  # Notify User the computer's response and game's result
    return result # Return You Win!, you lose!, and Draw!


def program(win=0,lose=0,draw=0):   # this program prompt the interface and gather user input \ the input are the game score default at zero
    user_entry = input(
        'Welcome to the game of Rock Paper Scissor!, \nPlease enter: "Rock", "Paper", or "Scissor"\nYour input:') #Initalization Prompt

########################## Initial Program Settings *****************************************
    valid_answer = ["Rock", "Paper", "Scissor"] # list of expected response from user
    hist_win_score =win       #Track the historical wins, takes default value for first round
    hist_lose_score =lose     #Track the historical loses, takes default value for first round
    hist_draw_score =draw     #Track the historical draws, takes default value for first round
    win_flag=0                #Indentifier if this round wins, reset to zero for first or new round
    lose_flag=0               #Indentifier if this round loses, reset to zero for first or new round
    draw_flag=0               #Indentifier if this round draws, reset to zero for first or new round
############################################################################################

    while not user_entry in valid_answer:   # User input validation
        user_entry = input('Please Enter a Valid input from "Rock", "Paper", or "Scissor":') # recollect input if user provided unexpected input
    ##### Executing the following when user input is satisfied #########################
    else:
        tracker = rpc(user_entry)       #Run the rock paper scissor function to determine user's winning status
        if tracker == "You Win!":
            hist_win_score += 1         # if user wins update the historical score
            win_flag=1                  # Flag this round is the winning round
        elif tracker == "you lose!":
            hist_lose_score += 1        # if user loses update the historical score
            lose_flag = 1               # Flag this round is the losing round
        elif tracker == "Draw!":
            hist_draw_score += 1        # if user draws update the historical score
            draw_flag=1                 # Flag this round is the drawing round
    #### This Round Ends, now we aak users if they want to play more ##############################

        try_again = input('Would you like to try again? (Enter "Yes" or "No"):')   # Ask users if they want to play more Yes or NO
        valid_answer2 = ["Yes", "No"]               #Expected answer from users
        while not try_again in valid_answer2:
            try_again = input('Please Specify "Yes" or "No":')   #recollect input if user provided unexpected input
        ###### Executing the following when user input is satisfied ###################
        else:
            if try_again == "Yes":  #User wants to play more
                output=program(hist_win_score,hist_lose_score,hist_draw_score) #initalize next round with updated historical scores a.k.a current sores as inputs
                win = output[0]       #collect the next round's winning score, (the score is cumulative)
                lose = output[1]      #collect the next round's losing score, (the score is cumulative)
                draw = output[2]      #collect the next round's drawing score, (the score is cumulative)


            elif try_again == "No": #User doesn't want to play more
                print("Thanks for playing, hope you had a good time!") #Ending Comments
                win += win_flag       #Update the scoreboard based on this rounds outcome
                lose += lose_flag     #Update the scoreboard based on this rounds outcome
                draw += draw_flag     #Update the scoreboard based on this rounds outcome

    lst = [win, lose, draw]       # Function's final output is the cumulative scoreboard

    return lst


def scoreboard():   #The function to call out the program and prints the final score
    final_score=program()      # The final score will be the cumulative scoreboard as result of running the program
    print("Your Final Score: Win:",final_score[0],"| Lose:",final_score[1],"| Draw:",final_score[2]) # Print out the score




scoreboard()  #Call out the scoreboard










