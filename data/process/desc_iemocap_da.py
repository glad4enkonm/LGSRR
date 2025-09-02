import os
import csv
from tqdm import tqdm
import pandas as pd

from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init

if __name__ == "__main__":

    raw_video_path = 'Dataset/IEMOCAP-DA/video_path'
    data_path = 'data/data_text/IEMOCAP-DA/train.tsv'
    video_caption_path = 'results/videollama2_iemocap-da.tsv'

    model_path = 'VideoLLaMA2-7B-16F'
    
    disable_torch_init()
    modal = 'video'
    model, processor, tokenizer = model_init(model_path)

    df = pd.read_csv(data_path, delimiter='\t')
    row_count = len(df)
    row_count = row_count + 1

    with open(video_caption_path, 'w', newline='') as output_file:
        writer = csv.writer(output_file, delimiter='\t')
        writer.writerow(['Dialogue_id', 'Utterance_id', 'Video_Caption'])

        with open(data_path, 'r') as f:
            data = csv.reader(f, delimiter="\t")
            for i, line in tqdm(enumerate(data), total=row_count, desc="Processing videos"):
                if i == 0:
                    continue
            
                index = line[0]
                video_path = os.path.join(raw_video_path,f'{index}.mp4')

                text = line[1]
                instruct = text + \
                            """
                            Generate detailed descriptions that help identify the speaker's intent. Please combine video and text to describe from the following perspectives:
                            1. **Speakers' Actions**: Provide a detailed account of the actions and movements of the speakers. Focus on gestures, posture, and any physical interactions between the characters.
                            2. **Facial Expressions**: Describe the emotions displayed by the speakers. Include details about their facial expressions, tone of voice, and any changes in mood throughout the video.
                            3. **Interaction with Others**: Detail the interactions between the speakers and other individuals in the scene. Note how they communicate, react to each other, and any visible signs of their relationship (e.g., familiarity, tension, formality).
                            Focus on these aspects to create a comprehensive description that would aid in recognizing the intentions behind the speakers' actions and words.
                            """
                
                response = mm_infer(processor[modal](video_path), instruct, model=model, tokenizer=tokenizer, do_sample=False, modal=modal)

                writer.writerow([line[0], response])