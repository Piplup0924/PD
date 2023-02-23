import pandas as pd

filename = "/home/hutu/PersonaDataset/PD/friends-personality/CSV/friends-personality.csv"
df = pd.read_csv(filename)

texts = df['text'].tolist()
characters = df['character'].tolist()

dialogues = []
for text in texts:
    dialogue = text.split("<br><br>")
    target_user = dialogue[0].split(" for ")[1][:-4]
    speakers = {}
    processed_dialogue = []
    for utterance in dialogue[1:-1]:
        if utterance[:3] != "<b>":
            continue
        speaker, utterance = utterance.split("</b>: ")
        speaker = speaker[3:]
        # if speaker == target_user:
        #     speaker = 0
        # else:
        #     if speaker not in speakers:
        #         speakers[speaker] = len(speakers) + 1
        #     speaker = speakers[speaker]
        processed_dialogue.append((speaker, utterance))
    dialogues.append(processed_dialogue)

print("gg")