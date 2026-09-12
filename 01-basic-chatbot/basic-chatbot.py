from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

load_dotenv()

model = ChatMistralAI(
    model = "open-mistral-7b",
    temperature= 0.7
)
print("-"*8,"Start Chat With Chatbot (Type 0 to exit)","-"*8)

print("Press 1 for Happy Agent")
print("Press 2 for Sad Agent")
print("Press 3 for Angry Agent")

mode_input = int(input("Enter Mode: "))
if mode_input == 1:
    mode = "You are a Happy chat Agent. Response user in happy mode"
elif mode_input == 2:
    mode = "You are a Sad chat Agent. Response user in Sad mode"
elif mode_input == 3:
    mode = "You are a Angry chat Agent. Response user in angry mode" 




while True:
    user_input = input("You: ")
    if user_input == "0":
        break
    message = [
    SystemMessage(content = mode)
]
    message.append(HumanMessage(user_input))

    response = model.invoke(message)
    message.append(AIMessage(response.content))

    print("Bot: ",response.content)