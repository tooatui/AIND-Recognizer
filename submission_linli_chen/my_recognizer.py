import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    # print(test_set.wordlist)
    # print(test_set.get_all_Xlengths())

    for index in range(0, len(test_set.get_all_Xlengths())):
      X, length = test_set.get_item_Xlengths(index)
      best_score = float("-inf")
      prob_dict = {}
      guess = ""

      # loop through each word model and find the best matched
      for word, model in models.items():
        try:
          logL = model.score(X, length)
          prob_dict[word] = logL

          if logL > best_score:
            best_score = logL
            guess = word
        except:
          pass

      probabilities.append(prob_dict)
      guesses.append(guess)

    return probabilities, guesses
