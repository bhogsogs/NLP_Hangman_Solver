# Hangman Solver

# Introduction

Hangman is a classic word-guessing game typically played between two players where one player thinks of a word and the other player tries to guess it by suggesting letters within a certain number of attempts. 

Hangman consists of essentially four main steps, Word Selection, Displaying Hidden Word, Guessing Letters, Game Termination. The goal of this problem statement was to develop an algorithm/model capable of playing the Hangman game effectively with a good score, that is, it had to suggest the next letter to guess based on the current state of the partially revealed word. The implementation should be able to handle unseen words during testing and provide accurate suggestions to maximize the chances of winning the game.

The dataset provided to us consisted of 50,000 english words, and we were not allowed to incorporate any other data source in our model training procedure. It is important to note that as a baseline, the statistical modeling approach is able to achieve an accuracy of 18% while simulating this game.

# Approach (MLM 2-Gram Model)

This model is essentially a combination of a statistical and a language model. Essentially, we observed that making the first prediction accurately with solely a language model can lead to low accuracy value, at the same time, relying on the frequency values of alphabets at the initial stages can prove to enhance the accuracy of our model. Keeping this in mind, we adopted a two step approach:-

For the first alphabet, our model will make the guess based on the letter frequency in the set of all the words of the same length in the training set. This step will keep going on until the algorithm has made a valid guess.

Once the first alphabet has been correctly guessed, our model calculates the possible 2-grams starting with this alphabet, and picks the most frequent one. If the model is unable to find a matched 2-gram, this is where the language model comes into play. Our LSTM Model, trained on a Masked Language Modeling Task, is used to make predictions. Essentially, we constructed an LSTM model with 3 layers trained on a masked language modeling task on the training corpus. We use this if model to make predictions  we are unable to find any valid 2-gram. The input to the model is the soft-encoded form of the currently “masked” hangman word, and it generates a character index as the output, from which the character is retrieved using a hash dictionary.

<p align="center">
   <img src="https://github.com/mbappeenjoyer/NLP_Hangman_Player/assets/134948011/91873cd9-c7b5-4cf0-b940-5c419fb062cd" width=900 height=300>
</p>

# Results

The MLM-2-Gram model achieved an impressive accuracy of 93.62% on an evaluation set comprising 5,000 words from the training corpus. The LSTM model was pretrained on a masked language modeling task, where it achieved a best case Cross Entropy Loss Score of 3.21 after running for 500 epochs. However, on a more challenging test set of 50,000 words, the model's accuracy was 43.52%.  
