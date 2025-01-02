import ollama

# Open a file to store input-output pairs
filename = input("enter the filename to store chat log: ")
filename += ".txt"
with open(filename, "a") as log_file:
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat...")
            break

        # Call the model to generate a response
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {
                    "role": "user",
                    "content": user_input,
                }
            ]
        )

        # Extract the model's response
        model_response = response['message']['content']
        print(f"Llama3: {model_response}")

        # Write input-output pairs to the file
        log_file.write(f"{user_input} {model_response}\n")
