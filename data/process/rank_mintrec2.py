from openai import OpenAI
from tqdm import tqdm,trange
import csv

client=OpenAI(api_key="")

SYSTEM_PROMPT = "You are a helpful and intelligent video comprehension system. I will provide you with the intent of the video. I will also provide description of the video from three different aspects along with the spoken text. You need to try to rank these three aspects according to their contribution to the intent of the video."
USER_PROMPT_1 = "Are you clear about your role?"
ASSISTANT_PROMPT_1 = "Sure, I'm ready to help you with your video comprehension task to generate 3 sorted aspects based on their contribution to the intent. I will not take the original order into account when generating the sorted aspects. I will rank the three aspects and not the possible intentions. I will only generate the three ranked aspect labels. I will not include any descriptions, explanations or introductions in my response. My answer will begin with the first of the ranked aspect labels. Please provide me with the necessary information to get started."
GUIDELINES_PROMPT="""
    Three aspects not in any particular order: Speakers' Actions, Facial Expressions, Interaction with Others
    Intent:<intent>
    Spoken text: <text>
    Three descriptions not in any particular order:
    **Facial Expressions**: <expression>
    **Interaction with Others**: <inter>
    **Speakers' Actions**: <actions>
"""

def openai_chat_completion_response(final_prompt):
    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_1},
        {"role": "assistant", "content": ASSISTANT_PROMPT_1},
        {"role": "user", "content": final_prompt}
    ])
    return response.choices[0].message.content

def prompt_wrap(intent,text,actions,expression,inter):
    final=GUIDELINES_PROMPT.replace("<actions>",actions)
    final=final.replace("<expression>",expression)
    final=final.replace("<inter>",inter)
    final=final.replace("<text>",text)
    final=final.replace("<intent>",intent)
    return final

def rank_response(sentence,words,dia,utt):
    index_dict={word: sentence.find(word) for word in words}
    sorted_words=sorted(words,key=lambda x: index_dict[x])
    return sorted_words

def generate_rank(num,writer,start=0):
    lines=[]
    lines_rank=[]
    # train_desc.tsv is the combination of train.tsv(the original data) and videollama2_mintrec2.tsv(the description data generated in CoT-Step 2).
    # for each video, train_desc.tsv contains the dialogue_id, utterance_id, intent, text and three generated descriptions in order.
    desc_path="data/data_desc/mintrec2.0/train.tsv"
    with open(desc_path,'r',encoding='utf-8') as desc_file:
        reader=csv.reader(desc_file,delimiter='\t')
        for line in reader:
            lines.append(line)
    for i in tqdm(range(start,num)):
        dia=lines[i+1][0]
        utt=lines[i+1][1]
        intent=lines[i+1][2]
        text=lines[i+1][3]
        actions=lines[i+1][4]
        expression=lines[i+1][5]
        inter=lines[i+1][6]

        final_prompt=prompt_wrap(
            intent=intent,
            text=text,
            actions=actions,
            expression=expression,
            inter=inter
        )
        response=openai_chat_completion_response(final_prompt)
        
        line_rank=rank_response(response,["Speakers' Actions","Facial Expressions","Interaction with Others"],dia,utt)
        line_rank.insert(0,utt)
        line_rank.insert(0,dia)
        lines_rank.append(line_rank)
        writer.writerow(line_rank)

with open("data/data_rank/rank_mintrec2_train.tsv",'w',newline='', encoding='utf-8') as file:
    writer=csv.writer(file,delimiter='\t')
    writer.writerow(["Dia","Utt","Rank_First","Rank_Second","Rank_Third"])
    num = 6165 # numbers of samples in the training set
    generate_rank(num,writer)