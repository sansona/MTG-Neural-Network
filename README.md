# MTG-Neural-Network

Goal: develop a neural network to predict the winner of a game of Magic the Gathering using data about the decks.

The program gathers data from two .csv files (blanks included)- one filled with match data for use as training data and the second with parameters about the decks being used. The neural network can be trained to learn about any number of decks for any number of players in a given game by editing lines 7-15 in ```mtg_nn.py```.


The files included are as such:

  - mtg_nn.py is where the neural network itself is located. Functionality for training and predictions are here.
  - param_vec.py handles the interception and formatting of data from the .csvs
  - activations.py contains the activation functions used in the neural network
  - weights.py handles initializing and updating the weights of the neural network
  - main.py for running the entirety of the program
  
 Additionally, two blank .csvs (DeckParameters.csv and TrainingData.csv) that were used for the training of the original network are included, but any .csvs will work so long as the appropriate parameters are changed in ```mtg_nn.py```


To train the network, open main.py and set ```test=False```. To predict the winner of a game, set ```test=True```. 

Example prediction:

```python main.py

Enter the name of the first deck:
Alesha, Who Smiles at Death

Enter the name of the second deck:
Brion Stoutarm

Enter the name of the third deck:
Ghoulcaller Gisa

The winner is predicted to be: Ghoulcaller Gisa with a confidence of [0.98]
```

To be improved upon further iterations (subtle ML joke):

  - one needs a very large dataset for training if appreciable accuracy is to be expected. Unfortunately, games of Magic     often take a long time which makes obtaining large datasets difficult. With small training sets, a high level of overfitting   occurs and is effectively unavoidable (unless much more data is available)
  - the unpacking and formatting of data can be improved upon. As of now, there are several steps that involve converting       back and forth between numpy arrays and python lists.
