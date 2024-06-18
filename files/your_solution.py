from model import Word2Batch, RNN_model
import torch
import collections
import pandas as pd
import numpy as np

char_to_index = {chr(i): i - 96 for i in range(97, 123)}
char_to_index.update({'_': 27})
char_to_index.update({'-': 0})
index_to_char = {i: char for char, i in char_to_index.items()}


def load_model(model_path):
    model = RNN_model(input_size=len(char_to_index), hidden_size=10)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    return model

def build_dictionary(dictionary_file_location):
    text_file = open(dictionary_file_location, "r")
    full_dictionary = text_file.read().splitlines()
    text_file.close()
    return full_dictionary

def gen_condition(word):
    tmp = {i: word[i] for i in range(len(word)) if word[i] != "_"}
    condition = {}
    for key, val in tmp.items():
        if val not in condition:
            condition[val] = [key]
        else:
            condition[val].append(key)
    return condition

def init_df(dictionary):
    group_by_length = collections.defaultdict(list)
    for word in dictionary:
        group_by_length[len(word)].append(word)

    res = {}
    for key in group_by_length.keys():
        word_list = group_by_length[key]
        tmp = pd.DataFrame([list(word) for word in word_list])
        tmp.columns = [chr(i + 97) for i in range(tmp.shape[1])]
        res[key] = tmp
    return res

def find_by_gram(all_gram, pre=None, suff=None):
    selected_gram = []
    for key, val in all_gram.items():
        if (pre is not None) and (key[0] == pre):
            selected_gram.append((key[1], val))
        if (suff is not None) and (key[1] == suff):
            selected_gram.append((key[0], val))

    res = {}
    for letter, freq in selected_gram:
        if letter not in res:
            res[letter] = freq
        else:
            res[letter] += freq
    final_res = [(key, val) for key, val in res.items()]
    return sorted(final_res, key=lambda x: x[1], reverse=True)

def gen_n_gram(word, n):
    n_gram = []
    for i in range(n, len(word)+1):
        if word[i-n:i] not in n_gram:
            n_gram.append(word[i-n:i])
    return n_gram
def init_n_gram(n):
    n_gram = {-1:[]}
    for word in full_dictionary:
        single_word_gram = gen_n_gram(word, n)
        if len(word) not in n_gram:
            n_gram[len(word)] = single_word_gram
        else:
            n_gram[len(word)].extend(single_word_gram)
        n_gram[-1].extend(single_word_gram)
    res = {}
    for key in n_gram.keys():
        res[key] = collections.Counter(n_gram[key])
    return res

full_dictionary_location = "training.txt"
full_dictionary = build_dictionary(full_dictionary_location)
full_dictionary_common_letter_sorted = collections.Counter("".join(full_dictionary)).most_common()
freq_by_length = init_df(full_dictionary)
n_gram = init_n_gram(2)
current_dictionary = []
history_condition = []
model = load_model("model.pth")

def freq_from_df(df):
    key, cnt = np.unique(df.values, return_counts=True)
    freq = [(k, val) for k, val in zip(key, cnt)]
    return sorted(freq, key=lambda x: x[1], reverse=True)

def update_df(df, condition):
    if len(condition) == 0:
        return df

    for letter, idx in condition.items():
        query = ""
        for i in range(df.shape[1]):
            col = df.columns.values[i]
            if i in idx:
                query += "{} == '{}' and ".format(col, letter)
            else:
                query += "{} != '{}' and ".format(col, letter)
        query = query[:-5]
        new_df = df.query(query)
        df = new_df.copy()
        del new_df
    return df


def encode_obscure_words(word):
    word_idx = [ord(i) - 97 if i != "_" else 26 for i in word]
    obscured_word = np.zeros((len(word), 27), dtype=np.float32)
    for i, j in enumerate(word_idx):
        obscured_word[i, j] = 1
    return obscured_word


def suggest_next_letter_sol(word,guessed_letters): 
    all_words = freq_by_length[len(word)]
    all_gram = n_gram[-1]
    new_condition = gen_condition(word)

    if len(history_condition) != 0 and new_condition != history_condition[-1]:
        history_condition.append(new_condition)

    all_words = update_df(all_words, new_condition)
    freq = freq_from_df(all_words)
    for i in range(len(freq)):
        if freq[i][0] not in guessed_letters:
            return freq[i][0]

    for i in range(len(word)):
        if word[i] == "_":
            if (i == 0) or (word[i-1] == "_"):
                guess = find_by_gram(all_gram, pre=None, suff=word[i+1])
            elif (i == len(word) - 1) or (word[i+1] == "_"):
                guess = find_by_gram(all_gram, pre=word[i-1], suff=None)
            else:
                guess = find_by_gram(all_gram, pre=word[i-1], suff=word[i+1])
            break

    for i in range(len(guess)):
        if guess[i][0] not in guessed_letters:
            return guess[i][0]
    # if we run out of 2-gram, use LSTM model to predict! 
    guessed_multi_hot = np.zeros(26, dtype=np.float32)
    for letter in guessed_letters:
        idx = ord(letter) - 97
        guessed_multi_hot[idx] = 1.0
    
    model.eval()
    masked_indices = torch.tensor([char_to_index[c] for c in word])
    masked_indices = masked_indices.unsqueeze(0)  
    output = model(masked_indices)
    char_ind = np.argmax(output[0].detach().numpy())
    pred_letter = index_to_char[char_ind]
    return pred_letter
