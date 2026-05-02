import os
from dotenv import load_dotenv
from groq import Groq
import json

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def get_weather(city):
    return "12 Degree Cel"


system_prompt = """
You are a helpful AI Assistant specialized in resolving user queries.

You work in steps:
1. plan
2. action
3. observe
4. output

For the given user query:
- Plan the steps
- Choose the correct tool
- Perform action
- Wait for observation
- Then return final output

Rules:
- Follow the Output JSON Format strictly
- Perform ONLY one step at a time
- Wait for next input after each step

Output JSON Format:
{
    "step": "plan | action | observe | output",
    "content": "string",
    "function": "function name (only for action)",
    "input": "input for function"
}

Available Tools:
- get_weather(city): Returns weather of a city

Example:
User: What is the weather of New York?

Step 1:
{ "step": "plan", "content": "User is asking for weather of New York" }

Step 2:
{ "step": "action", "function": "get_weather", "input": "New York" }

Step 3:
{ "step": "observe", "output": "12°C" }

Step 4:
{ "step": "output", "content": "Weather in New York is 12°C" }
"""

message = [
   {"role":"system", "content":system_prompt},
]

user_query = input('> ')
message.append({"role":"user", "content":user_query})

while True:
    response = client.chat.completions.create(
         model="llama-3.1-8b-instant",
         response_format={"type":"json_object"},
         messages=message
    )
    parsed_output = json.loads(response.choices[0].message.content)
    message.append({"role":"assistant", "content":json.dumps(parsed_output)})


print(respose.choices[0].message.content)