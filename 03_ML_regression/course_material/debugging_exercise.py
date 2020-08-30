import json

def is_answer(node):
    return len(node) == 1

f = open('questions.json')
content = f.read()
node = json.loads(content)
#node = {"text": "Is it a snake?", "yes": {"text": "It is a Python!"}, "no": {"text": "It is not a Python"}}

#with open('questions.json') as f:
 # node = json.load(f)

finished = False

while not finished:
    print(node['text'])
    if is_answer(node):
        finished = True
    else:
        answer = input()
        if answer.lower() in ['yes', 'y']:
            node = node['yes']
        else:
            node = node['no']