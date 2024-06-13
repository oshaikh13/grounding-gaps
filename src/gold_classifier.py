
import json
import openai 
import argparse
from tqdm import tqdm
import time
import random
random.seed(420)

## Setup OpenAI API

# with open('api_key.txt','r') as f:
#     openai.api_key = [line.rstrip('\n') for line in f][0]

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', help='input-data')
parser.add_argument('--output-path', help='input-data')
parser.add_argument('--prompt-path', help='input-data')
parser.add_argument('--model', help='input-data', default="gpt-4")
parser.add_argument('--limit', help='input-data', type=int)
parser.add_argument('--cot', default=False, action="store_true")
parser.add_argument('--rev-test', default=False, action="store_true")
parser.add_argument('--skip-test', type=int)

args = parser.parse_args()

## Setup datapath loading

with open(args.data_path + "/test_data.json", "r") as f:
    convos = json.load(f)

with open(args.prompt_path, "r") as f:
    classification_prompt = f.read()

turn_length = []
for elem in convos:
    turn_length.append(len(elem["messages"]))

min_turns = min(turn_length)
print(f"EXTRACTED MIN TURNS: {min_turns}")

for elem in convos:
    elem["messages"] = elem["messages"][:min_turns]

random.shuffle(convos)

if args.cot:
    print("USING COT")

# sample limit random from convos
if args.limit:
    convos = convos[:args.limit]

if args.skip_test:
    print("SKIPPING TEST")
    if args.rev_test:
        convos = convos[:args.skip_test]
    else:
        convos = convos[args.skip_test:]

try:
    with open(args.output_path, "r") as f:
        output_map = json.load(f)
except:
    print("LOG: OUTPUT MAP NOT FOUND")
    output_map = {}

def openai_req(classification_prompt, past_messages, get_history=False):
    
    message_text = ""
    for message in past_messages:
        message_text += f"{message['role']}: {message['content']}\n"

    first_message = past_messages[0]
    message_text += f"\nAnnotated:\n\n{first_message['role']}: {first_message['content']}\nLabel:"

    print(f"{classification_prompt}{message_text}")

    history = [{ 
        "role": "assistant",
        "content": f"{classification_prompt}{message_text}"
    }]

    running_history = [{ 
        "role": "assistant",
        "content": f"{classification_prompt}{message_text}"
    }]

    collected_chunks = []
    collected_messages = []

    if get_history:
        return history

    response = None
    while True:
    # TODO: convert this into text-davinci-completion APIs
        try:
            start_time = time.time()
            response = openai.ChatCompletion.create(
                engine=args.model,
                n=1, 
                temperature=0,
                messages=running_history,
                stream=True
            )

        # iterate through the stream of events
            for chunk in response:
                chunk_time = time.time() - start_time  # calculate the time delay of the chunk
                if len(chunk["choices"]) == 0: continue
                collected_chunks.append(chunk)  # save the event response
                chunk_message = chunk['choices'][0]['delta']  # extract the message
                collected_messages.append(chunk_message)  # save the message
                print(f"Message received {chunk_time:.2f} seconds after request: {chunk_message}")  # print the delay and text
                curr_message = ''.join([m.get('content', '') for m in collected_messages])
                split_messages = curr_message.split("Label:")
                print(f"SPLIT {len(split_messages)} PAST {len(past_messages)}")
                print(len(past_messages))
                if len(split_messages) > len(past_messages):
                    return history, curr_message

            # print the time delay and text received
            print(f"Full response received {chunk_time:.2f} seconds after request")
            full_reply_content = ''.join([m.get('content', '') for m in collected_messages])

            time.sleep(0.3)
            return history, full_reply_content
        except Exception as e:
            print("CAUGHT EXCEPTION")
            print(e)
            print(type(e).__name__)
            curr_message = ''.join([m.get('content', '') for m in collected_messages])
            # print(f"{classification_prompt}{message_text}")
            # running_history[0]["content"] = running_history[0]["content"] + curr_message
            if type(e).__name__ == "TypeError": 
                print("TypeError, returning None")
                return None
            if "InvalidRequestError" in type(e).__name__:
                print("SKIPPING CONVERSATION")
                return None
            if "timeout" in type(e).__name__.lower():
                print("TIMEOUT SKIP")
                return None
            
            collected_chunks = []
            collected_messages = []

            continue

for idx, convo in tqdm(enumerate(convos), total=len(convos)):
    
    if str(idx) in output_map and output_map[str(idx)] != None: 
        print("SKIPPING")
        # _, api_completion = output_map[str(idx)]
        # print(api_completion["choices"][0]["message"]["content"])
        continue

    if str(idx) in output_map and output_map[str(idx)] == None:
        print("NONE FIELD")
        print(str(idx))

    res = openai_req(classification_prompt, convo["messages"])
    
    if res == None: 
        output_map[str(idx)] = None
        continue

    _, api_completion = res

    output_map[str(idx)] = (_, {
        "choices": [{"message": {"content": api_completion}}]
    })

    # save file here
    with open(args.output_path, "w") as f:
        json.dump(output_map, f, indent=4)
    