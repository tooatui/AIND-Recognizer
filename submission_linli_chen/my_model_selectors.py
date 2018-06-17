import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # TODO implement model selection based on BIC scores
        best_bic_score = float("inf")
        best_n = self.n_constant

        # loop through all possible n_components and find the best
        for n in range(self.min_n_components, self.max_n_components + 1):
            # check the count of training
            try:
                hmm_model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                logL = hmm_model.score(self.X, self.lengths)

                N, f = self.X.shape
                p = n ** 2 + 2 * n * f - 1

                bic = -2 * logL + p * math.log(N)

                if bic < best_bic_score:
                    best_bic_score = bic
                    best_n = n
            except:
                pass

        # print("{0} best_n: {1}".format(self.this_word, best_n))
        return self.base_model(best_n)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # TODO implement model selection based on DIC scores
        best_dic_score = float("-inf")
        best_n = self.n_constant

        competing_words = [key for key in self.words.keys() if key != self.this_word]
        M = len(self.hwords)

        # loop through all possible n_components and find the best
        for n in range(self.min_n_components, self.max_n_components + 1):
            # check the count of training
            try:
                hmm_model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                logL = hmm_model.score(self.X, self.lengths)

                sum_competing_logL = 0
                # get the score for the rest of the words
                for word in competing_words:
                    X, lengths = self.hwords[word]
                    sum_competing_logL += hmm_model.score(X, lengths)

                dic = logL - (1 / M - 1) * sum_competing_logL

                if dic > best_dic_score:
                    best_dic_score = dic
                    best_n = n
            except:
                pass

        # print("{0} best_n: {1}".format(self.this_word, best_n))
        return self.base_model(best_n)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # TODO implement model selection using CV
        best_average_logL = float("-inf")
        best_n = self.n_constant

        # loop through all possible n_components and find the best
        for n in range(self.min_n_components, self.max_n_components + 1):
            # no way to fold
            if len(self.sequences) < 2:
                hmm_model = self.base_model(n)
                if hmm_model is not None:
                    logL = hmm_model.score(self.X, self.lengths)
                    if logL > best_average_logL:
                        best_average_logL = logL
                        best_n = n

            else:
                total_logL = 0
                # track the count of training
                count = 0            

                if (len(self.sequences) <= 3):
                    split_method = KFold(n_splits=len(self.sequences))
                else:
                    # by default the nn_splits = 3
                    split_method = KFold()

                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    # print("Train fold indices:{} Test fold indices:{}".format(cv_train_idx, cv_test_idx))  # view indices of the folds
                
                    X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                    X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences) 

                    try:
                        hmm_model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                                random_state=self.random_state, verbose=False).fit(X_train, lengths_train)
                        logL = hmm_model.score(X_test, lengths_test)
                        total_logL += LogL
                        count += 1
                    except:
                        pass


                if count > 0 and total_logL/count > best_average_logL:
                    best_average_logL = total_logL/count
                    best_n = n

        # print("{0} best_n: {1}".format(self.this_word, best_n))
        return self.base_model(best_n)