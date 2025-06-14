import ollama

# Open a file to store input-output pairs
filename = input("Enter the filename to store chat log: ")
filename += ".txt"
instruction_text = (
    "You will analyze a series of statements and assign two scores for each based on the specified axes. "
    "The X-axis represents economic ideology and ranges from -1 to 1. Negative scores on the X-axis indicate left-wing economic perspectives, "
    "which emphasize collective welfare, government regulation, wealth redistribution, and public ownership, while positive scores on the X-axis "
    "reflect right-wing economic perspectives, prioritizing free markets, minimal government intervention, privatization, and individual entrepreneurship. "
    "The Y-axis represents social ideology, also ranging from -1 to 1. Negative scores on the Y-axis indicate liberal perspectives, characterized by "
    "personal freedoms, openness to social change, reduced state control, and the protection of individual rights, while positive scores on the Y-axis "
    "represent authoritarian perspectives, emphasizing state control, law and order, adherence to traditional values, and limited individual freedoms for societal goals. "
    "For example, a statement advocating for universal healthcare funded through progressive taxation would score approximately -0.8 on the X-axis and -0.6 on the Y-axis, "
    "reflecting a left-wing economic perspective with moderately liberal social implications. A statement supporting the deregulation of financial markets and reduced corporate tax "
    "would score around 0.9 on the X-axis and 0 on the Y-axis, indicating a strongly right-wing economic position with neutral social implications. "
    "A statement calling for strict government surveillance to combat crime would score around 0 on the X-axis and 0.8 on the Y-axis, reflecting a neutral economic stance with a strongly authoritarian social perspective. "
    "A statement promoting gender equality and the legalization of same-sex marriage would score approximately 0 on the X-axis and -0.9 on the Y-axis, demonstrating a neutral economic stance and strongly liberal social perspective. "
    "Additional examples include a statement advocating for wealth redistribution through high taxes on the rich (-0.9, -0.2), the privatization of public schools (0.8, 0.2), "
    "government mandates for wearing uniforms in public schools (0.3, 0.6), support for environmental regulation of businesses (-0.7, -0.3), opposition to immigration (0, 0.7), "
    "promoting free college education funded by the state (-1, -0.4), and support for maintaining traditional family structures through policy incentives (0.2, 0.8). "
    "A statement proposing a ban on public protests for national security would score around 0.1 on the X-axis and 0.9 on the Y-axis, while one advocating for reducing military budgets to fund social welfare programs would score -0.8 on the X-axis and -0.6 on the Y-axis. "
    "For every statement, evaluate its economic and social implications independently and provide scores within the range of -1 to 1. Your reply must strictly adhere to the format: \"score on x axis, score on y axis\". "
    "Do not provide any additional explanation, context, or formatting—only the two scores in the specified format for each statement. "
    "If a statement is neutral, normal, a question, or has no clear political or ideological connotation, you must assign the score \"0, 0.\" Examples include \"The sky is blue,\" \"What is your favorite color?\" \"I enjoy painting,\" or \"The weather is pleasant today.\" "
    "These types of statements, which are descriptive, interrogative, or unrelated to political or ideological frameworks, must always receive the score \"0, 0.\" "
    "Under no circumstances should you return anything other than the score in the exact format: \"score on x axis, score on y axis.\" "
)

with open(filename, "a") as log_file:
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat...")
            break

        # Prepend instruction text and format user input
        formatted_input = f"{instruction_text} Text: {user_input}"

        # Call the model to generate a response
        response = ollama.chat(
            model="llama3.1",
            messages=[
                {
                    "role": "user",
                    "content": formatted_input,
                }
            ]
        )

        # Extract the model's response
        model_response = response['message']['content']
        print(f"Llama3: {model_response}")

        # Write input-output pairs to the file
        log_file.write(f"{user_input} {model_response}\n")
