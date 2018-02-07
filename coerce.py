import pickle
import numpy as np
import pandas as pd
import midi
import random

hm_lines = 100000

def read_notes(fn):
    print("Reading:", fn)
    notes = []
    f = midi.read_midifile(fn)
    for track in f:
        for note in track:
            if isinstance(note, midi.NoteOnEvent):
                notes.append(note.data)
    print("Read:", len(notes), "notes")
    return notes

# read_notes('bach/bach_846.mid')

def sample_handling(files): #, notes, classification):
    def _sample_handling(sample):
        featureset = []
        notes = read_notes(sample)
        for note1, note2 in zip(notes, notes[1:]):
            featureset.append([note1, note2])
        return featureset
    ret = [f for file in files for f in _sample_handling(file)]
    return ret

def create_feature_sets_and_labels(files, test_size = 0.1):
    _features = np.array(sample_handling(files))
    testing_size = int(test_size*len(_features))
    classes = np.unique(_features[:,0][:-testing_size])
    n_classes = len(classes)
    features = []
    for f in _features:
        result = np.zeros(n_classes)
        result[np.where(classes==f[1])[0]] = 1
        features.append([f[0], result])
    
    random.shuffle(features)
    features = np.array(features)

    testing_size = int(test_size*len(features))

    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])
    return train_x, train_y, test_x, test_y, n_classes
    
files = ['samples/bach/bach_846.mid', 'samples/bach/bach_847.mid', 'samples/bach/bach_850.mid']
if __name__ == '__main__':
    train_x, train_y, test_x, test_y, n_classes = create_feature_sets_and_labels(files)
    # if you want to pickle this data:
    with open('note_features.pickle','wb') as f:
        pickle.dump([train_x, train_y, test_x, test_y, n_classes],f)
