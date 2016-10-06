import os, re
import numpy as np

def load_texts():
    text = ''
    topic_folders = os.listdir('bbc')
    print(topic_folders)
    for topic_folder in topic_folders:
        if os.path.isdir(os.path.join('bbc', topic_folder)): # consider only folders
            print(topic_folder)
            topic_texts = os.listdir(os.path.join('bbc', topic_folder))

            for topic_text in topic_texts:
                # print(topic_text)
                with open(os.path.join('bbc', topic_folder, topic_text), 'r') as reader:
                    # skip first line (heading)
                    reader.readline()
                    line = reader.readline().decode("ascii","ignore").encode("ascii")
                    while line:
                        if line.strip() is not '':
                            line = line.strip()
                            # line = re.sub(r'\.([a-z]+)', r'. \1', line)
                            text += " " + line
                        line = reader.readline().decode("ascii","ignore").encode("ascii")


    return text


def load_glove():
    GLOVE_FOLDER = '/home/arahusky/Desktop/glove'
    FILENAME = 'glove.6B.50d.txt'
    glove_mapping = {}
    with open(os.path.join(GLOVE_FOLDER, FILENAME), 'r') as reader:
        line = reader.readline()
        while line:
            chunks = line.split(' ')
            glove_mapping[str(chunks[0])] = np.array(map(float, chunks[1:]))
            # print(str(chunks[0]))
            line = reader.readline()

    return glove_mapping
