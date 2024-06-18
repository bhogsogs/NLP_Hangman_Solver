import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.distributed as dist
import collections
CUDA = torch.cuda.is_available()


def show_game(original_word, guesses, obscured_words_seen):
    print('Hidden word was "{}"'.format(original_word))

    for i in range(len(guesses)):
        word_seen = ''.join([chr(i + 97) if i != 26 else ' ' for i in obscured_words_seen[i].argmax(axis=1)])
        print('Guessed {} after seeing "{}"'.format(guesses[i], word_seen))

def list2tensor(arr):
    arr = np.array(arr)
    return torch.from_numpy(arr)


class Word2Batch:
    def __init__(self, model, word, lives=6):
        self.origin_word = word
        self.guessed_letter = set()  
        self.word_idx = [ord(i)-97 for i in word]
        self.remain_letters = set(self.word_idx)
        self.model = model
        self.lives_left = lives
        self.guessed_letter_each = []

        self.obscured_word_seen = [] 
        self.prev_guessed = []  
        self.correct_response = []  

    def encode_obscure_word(self):
        word = [i if i in self.guessed_letter else 26 for i in self.word_idx]
        obscured_word = np.zeros((len(word), 27), dtype=np.float32)
        for i, j in enumerate(word):
            obscured_word[i, j] = 1
        return obscured_word

    def encode_prev_guess(self):
        guess = np.zeros(26, dtype=np.float32)
        for i in self.guessed_letter:
            guess[i] = 1.0
        return guess

    def encode_correct_response(self):
        response = np.zeros(26, dtype=np.float32)
        for i in self.remain_letters:
            response[i] = 1.0
        response /= response.sum()
        return response

    def game_mimic(self):
        obscured_words_seen = []
        prev_guess_seen = []
        correct_response_seen = []

        while self.lives_left > 0 and len(self.remain_letters) > 0:
            obscured_word = self.encode_obscure_word()
            prev_guess = self.encode_prev_guess()

            obscured_words_seen.append(obscured_word)
            prev_guess_seen.append(prev_guess)
            obscured_word = torch.from_numpy(obscured_word)
            prev_guess = torch.from_numpy(prev_guess)
            if CUDA:
                obscured_word = obscured_word.cuda()
                prev_guess = prev_guess.cuda()

            self.model.eval()
            guess = self.model(obscured_word, prev_guess)  
            guess = torch.argmax(guess, dim=2).item()
            self.guessed_letter.add(guess)
            self.guessed_letter_each.append(chr(guess + 97))

            correct_response = self.encode_correct_response()
            correct_response_seen.append(correct_response)

            if guess in self.remain_letters:  
                self.remain_letters.remove(guess)

            if correct_response_seen[-1][guess] < 0.0000001:  # which means we made a wrong guess
                self.lives_left -= 1
        obscured_words_seen = list2tensor(obscured_words_seen)
        prev_guess_seen = list2tensor(prev_guess_seen)
        correct_response_seen = list2tensor(correct_response_seen)
        return obscured_words_seen, prev_guess_seen, correct_response_seen

char_to_index = {chr(i): i - 96 for i in range(97, 123)}
char_to_index.update({'_': 27})
char_to_index.update({'-': 0})
index_to_char = {i: char for char, i in char_to_index.items()}

class RNN_model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3):
        super(RNN_model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        # Define LSTM
        self.lstms = nn.ModuleList([nn.LSTM(hidden_size, hidden_size, batch_first=True) for _ in range(num_layers)])
        
        # Output layer
        self.fc = nn.Linear(hidden_size, input_size)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input):
        # Embed input
        embedded = self.embedding(input)
        
        # Forward pass through LSTM layers
        output = embedded
        for lstm in self.lstms:
            output, _ = lstm(output)
        
        # Output layer
        output = self.fc(output[:, -1, :])  
        output = self.softmax(output)
        return output

def gen_n_gram(word, n):
    n_gram = []
    for i in range(n, len(word)+1):
        if word[i-n:i] not in n_gram:
            n_gram.append(word[i-n:i])
    return n_gram

def init_n_gram(n):
    n_gram = {}
    full_dictionary = ["apple", "hhh", "genereate", "google", "abc", "googla"]
    for word in full_dictionary:
        single_word_gram = gen_n_gram(word, n)
        print(word, single_word_gram)
        if len(word) not in n_gram:
            n_gram[len(word)] = single_word_gram
        else:
            n_gram[len(word)].extend(single_word_gram)
    print(n_gram)
    res = {}
    for key in n_gram.keys():
        res[key] = collections.Counter(n_gram[key])
    return res

if __name__ == "__main__":
    word = "compluvia"
    print(init_n_gram(2))




