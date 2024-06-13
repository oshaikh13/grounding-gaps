
import json
import openai 
import copy
import argparse
from tqdm import tqdm
import time
import random
random.seed(420)
import sys

## Setup OpenAI API

# openai.api_type = "azure"
with open('api_key.txt','r') as f:
    openai.api_key = [line.rstrip('\n') for line in f][0]


parser = argparse.ArgumentParser()
parser.add_argument('--data-path', help='input-data')
parser.add_argument('--output-path', help='input-data')
parser.add_argument('--limit', help='input-data', type=int)
parser.add_argument('--model', help='model')
args = parser.parse_args()

get_completion = None

## Setup datapath loading

with open(args.data_path + "/system_prompts.json", "r") as f:
    system_prompts = json.load(f)

with open(args.data_path + "/test_data.json", "r") as f:
    convos = json.load(f)

# cap conversation length across dataset.
turn_length = []
for elem in convos:
    turn_length.append(len(elem["messages"]))

min_turns = min(turn_length)
print(f"EXTRACTED MIN TURNS: {min_turns}")

for elem in convos:
    elem["messages"] = elem["messages"][:min_turns]

random.shuffle(convos)

# sample limit random from convos
if args.limit:
    convos = convos[:args.limit]

try:
    with open(args.output_path, "r") as f:
        output_map = json.load(f)
except:
    print("LOG: OUTPUT MAP NOT FOUND")
    output_map = {}


def format_gpt_completion(history, simulated_role):
    system = history[0]
    rem = history[1:]

    curr_input = system["content"] + "\n\n"
    
    for k in rem:
        curr_input += f"{k['role']}: {k['content']}\n"

    curr_input += simulated_role + ":"
    return curr_input
    
# test
def openai_req(
        system_prompts, 
        context, 
        past_messages,
        system_ablation="standard",
    ):
    
    history = copy.deepcopy(past_messages)

    # identity for system role
    role_map = { "system" : "system" }

    simulated_role = list(system_prompts.keys())
    simulated_role.remove(past_messages[-1]["role"])
    other_role = past_messages[-1]["role"]
    simulated_role = simulated_role[0]

    role_map[simulated_role] = "assistant"
    role_map[past_messages[-1]["role"]] = "user"

    history = [{ 
        "role": "system",
        "content": f"{system_prompts[simulated_role][system_ablation]}" + 
            (f"{context[simulated_role]}" if context else "")
    }] + history

    req_messages = []
    for message in history:
        req_message = {}
        
        req_message["role"] = role_map[message["role"]]            
        req_message["content"] = message["content"]
        req_messages.append(req_message)

    response = None
    while True:
    # TODO: convert this into text-davinci-completion APIs
        try:
            
            if simulated_role == "seeker":
                return history, None
                
            if args.model in ["gpt-4", "gpt-3.5-turbo"]: 
                response = openai.ChatCompletion.create(model=args.model, n=1, messages=req_messages)
                time.sleep(0.3)
            else:
                
                curr_prompt = format_gpt_completion(req_messages, simulated_role)
                
                response = {
                    "choices": [
                        {
                            "text": ""
                        }
                    ]
                }
                
                while len(response["choices"][0]["text"].strip()) == 0:
                    response = openai.Completion.create(
                        model=args.model,
                        prompt=curr_prompt,
                        max_tokens=128,
                        logit_bias={"50256": -50, "198": -5, "628": -5},
                        stop=["\n"]
                    )
            
            return history, response
        except Exception as e:
            print("CAUGHT EXCEPTION")
            if type(e).__name__ == "TypeError": 
                print("SKIPPING CONVERSATION!")
                return None
            if "InvalidRequestError" in type(e).__name__:
                print("SKIPPING CONVERSATION")
                return None
            print("not skipping")
            print(type(e).__name__)
            print()
            if args.lora_path is None:
                time.sleep(30)
            continue

for idx, convo in tqdm(enumerate(convos), total=len(convos)):
    
    if str(idx) not in output_map: 
        output_map[str(idx)] = {}

    for i in tqdm(range(1, len(convo["messages"]) + 1)):
        if str(i) in output_map[str(idx)]: 
            continue 

        res = openai_req(system_prompts, convo["context"], convo["messages"][:i])
        
        if res is None:
            print("SKIPPING")
            output_map[str(idx)]["skipped"] = True
            break

        output_map[str(idx)][str(i)] = res

        # save file here
        with open(args.output_path, "w") as f:
            json.dump(output_map, f, indent=4)
    
