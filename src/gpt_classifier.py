
import json
import openai 
import argparse
from tqdm import tqdm
import time
import random
random.seed(420)
from collections import defaultdict

with open('api_key.txt','r') as f:
    openai.api_key = [line.rstrip('\n') for line in f][0]

parser = argparse.ArgumentParser()
parser.add_argument('--gpt-data-path', help='input-data')
parser.add_argument('--gold-cls-path', help='input-data')
parser.add_argument('--output-path', help='input-data')
parser.add_argument('--prompt-path', help='input-data')
parser.add_argument('--model', help='input-data', default="gpt-4")
parser.add_argument('--cot', default=False, action="store_true")
parser.add_argument('--skip-test', type=int)
args = parser.parse_args()

def api_req(history, stop_tokens=None):
    while True:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4", 
                n=1, 
                messages=history,
                temperature=0,
                stop=stop_tokens
            )
            time.sleep(1)
            return history, response
        except Exception as e:
            print("CAUGHT EXCEPTION")
            if type(e).__name__ == "TypeError": 
                print("TypeError, returning None")
                return None
            if "InvalidRequestError" in type(e).__name__:
                print("InvalidRequestError")
                return None
            print("not skipping")
            print(type(e).__name__)
            print()
            time.sleep(10)
            continue

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def process_labels(labels):
    parsed_labels = defaultdict(list)
    for k in labels:
        if labels[k] == None:
            continue
        request, response = labels[k]
            
        text_response = response["choices"][0]["message"]["content"]
        req_start = request[0]["content"].split("\n\n")[-1]
        # if "Explanation" in req_start:
        text_response =  req_start.split("Annotated:\n")[-1] + text_response

        all_messages = None
        for z in text_response.split("\n"):
            if z.strip() == "": 
                all_messages = [x.split("\n") for x in text_response.split("\n\n")]
                break
        
        if all_messages is None:
            print("NO NEWLINE SEP")
            all_messages = [x for x in chunks(text_response.split("\n"), 2)]

        for curr_message in all_messages:
            try:
                role, messsage = curr_message[0].split(":")[:2]
                role = role.strip()
                message = messsage.strip()
                label = curr_message[-1]
                
                q = int(k) + args.skip_test
                parsed_labels[str(q)].append({
                    "role": role,
                    "content": message,
                    "explanation": curr_message[1:-1] if "Explanation" in req_start else None,
                    "label": label
                })
            except Exception as e:
                print(e)
                print("PARSE_FAIL: " + str(curr_message))
                # pdb.set_trace()

    return parsed_labels

## Setup datapath loading

with open(args.gold_cls_path, "r") as f:
    parsed_labels = process_labels(json.load(f))

with open(args.gpt_data_path, "r") as f:
    gpt_convos = json.load(f)    

with open(args.prompt_path, "r") as f:
    classification_prompt = f.read()

try:
    with open(args.output_path, "r") as f:
        output_map = json.load(f)
except:
    print("LOG: OUTPUT MAP NOT FOUND")
    output_map = {}

print(len(gpt_convos))

if args.skip_test:
    print("SKIPPING TEST")
    for i in range(args.skip_test):
        gpt_convos.pop(str(i))

def openai_req(
    classification_prompt, 
    past_messages, 
    gpt_message,
    content_key = "content"
):
    message_text = ""
    for message in past_messages:
        message_text += f"{message['role']}: {message['content']}\n"

    message_text += f"{gpt_message['role']}: {gpt_message[content_key]}\n"

    message_text += "\nAnnotated:\n\n"  

    for message in past_messages:
        message_text += f"{message['role']}: {message['content']}\n{message['label']}\n\n"


    message_text += f"{gpt_message['role']}: {gpt_message[content_key]}\nLabel:"

    history = [{ 
        "role": "assistant",
        "content": f"{classification_prompt}{message_text}"
    }]

    ret = api_req(history, stop_tokens=["\n\n"])
    return ret

for idx in tqdm(parsed_labels, total=len(parsed_labels)):
    # pdb.set_trace()
    convo = parsed_labels[idx]
    if str(idx) in output_map: 
        print("CONTAINED")
        
        if len(output_map[idx]) > 15:
            continue

    output_map[str(idx)] = []
    for i in tqdm(range(1, len(convo) + 1)):


        # get gpt message and set ground truth role
        try:
            request, response = gpt_convos[str(idx)][str(i)]
            
 
            if response is None:
                response = { "choices": [{ "text": "None" }] }
            
            # backwards compatibility
            if "message" in response["choices"][0]:
                response["choices"][0]["text"] = response["choices"][0]["message"]
            
            if ":" in response["choices"][0]["text"]:
                response["choices"][0]["text"] = response["choices"][0]["text"].split(":")[0].replace("user", "")
                
            gpt_message = response["choices"][0]
            # TODO: support all roles
            gpt_message["role"] = parsed_labels[str(idx)][i]["role"]
            if gpt_message["role"] == "seeker": continue

            # don't classify supp
            if parsed_labels[str(idx)][i]["role"] == "seeker":
                gpt_label_res = [
                    None,
                    { "choices": [{ "message": { "content": "None" } }] }
                ]
            else:
                gpt_label_res = openai_req(
                    classification_prompt, 
                    parsed_labels[str(idx)][:i],
                    gpt_message=gpt_message,
                    content_key="text"
                )

            # print(classification_prompt)
            # print(gpt_label_res[1]["choices"][0])
            output_map[str(idx)].append({
                "gpt_message": gpt_message["text"],
                "gpt_label": gpt_label_res[1]["choices"][0]["message"]["content"],
                "ground_truth": parsed_labels[str(idx)][i]["content"],
                "ground_truth_label": parsed_labels[str(idx)][i]["label"],
                "role": parsed_labels[str(idx)][i]["role"]
            })

            with open(args.output_path, "w") as f:
                json.dump(output_map, f, indent=4)
    
        except Exception as e:
            print("FUCK!!!")
            print(e)
            continue
