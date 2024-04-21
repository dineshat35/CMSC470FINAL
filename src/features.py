# Jordan Boyd-Graber
# 2023
#
# Feature extractors to improve classification to determine if an answer is
# correct.

from collections import Counter
from math import log
from numpy import mean
import gzip
import json
import math
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


class Feature:
    """
    Base feature class.  Needs to be instantiated in params.py and then called
    by buzzer.py
    """

    def __init__(self, name):
        self.name = name

    def __call__(self, question, run, guess, guess_history):
        raise NotImplementedError("Subclasses of Feature must implement this function")


"""
Given features (Length, Frequency)
"""


class LengthFeature(Feature):
    """
    Feature that computes how long the inputs and outputs of the QA system are.
    """

    def __call__(self, question, run, guess, guess_history):
        # print("\n")
        # print("QUESTION ", question)
        # print("\n")
        # print("RUN ", run)
        # print("\n")
        # print("GUESS ", guess)
        # print("\n")
        # print("GUESS HISTORY ", guess_history)
        # print("\n")

        # How many characters long is the question?
        yield ("char", (len(run) - 450) / 450)

        # How many words long is the question?
        yield ("word", (len(run.split()) - 75) / 75)

        ftp = 0

        # How many characters long is the guess?
        if guess is None or guess == "":
            yield ("guess", -1)
        else:
            yield ("guess", log(1 + len(guess)))


class FrequencyFeature:
    def __init__(self, name):
        from eval import normalize_answer

        self.name = name
        self.counts = Counter()
        self.normalize = normalize_answer

    def add_training(self, question_source):
        import json

        with gzip.open(question_source) as infile:
            questions = json.load(infile)
            for ii in questions:
                self.counts[self.normalize(ii["page"])] += 1

    def __call__(self, question, run, guess, guess_history):
        yield ("guess", log(1 + self.counts[self.normalize(guess)]))


class PalindromeCountFeature(Feature):
    def __init__(self, name):
        self.name = name

    def count_palindromes(self, text):
        words = text.split()
        words = [word.strip(".,!?").lower() for word in words]
        return sum(word == word[::-1] for word in words)

    def __call__(self, question, run, guess, guess_history):
        palindrome_count = self.count_palindromes(run)
        yield ("count", palindrome_count)


class SingularPluralFeature:
    def __init__(self, name):
        self.name = name

    def calculate_feature_value(self, guess):
        if not guess:
            return -1
        last_word_guess = guess.split()[-1]
        if last_word_guess.endswith("s"):
            return 2
        else:
            return -1

    def __call__(self, question, run, guess, guess_history):
        feature_value = self.calculate_feature_value(guess)
        yield ("score", feature_value)


class DifficultyLevelFeature:
    def __init__(self, name):
        self.name = name
        self.difficulty_to_ordinal = {"MS": 1, "HS": 2, "national_high_school": 3, "easy_college": 4, "regular_college": 5, "College": 6, "Open": 7}

    def __call__(self, question, run, guess, guess_history):
        difficulty_level = question.get("difficulty")
        ordinal_value = self.difficulty_to_ordinal.get(difficulty_level, 3)
        yield ("level", ordinal_value)


class GuessBlankFeature(Feature):
    """
    Is guess blank?
    """

    def __call__(self, question, run, guess):
        yield ("true", len(guess) == 0)


class GuessCapitalsFeature(Feature):
    """
    Capital letters in guess
    """

    def __call__(self, question, run, guess):
        yield ("true", log(sum(i.isupper() for i in guess) + 1))


if __name__ == "__main__":
    """

    Script to write out features for inspection or for data for the 470
    logistic regression homework.

    """
    import argparse

    from params import (
        add_general_params,
        add_question_params,
        add_buzzer_params,
        add_guesser_params,
        setup_logging,
        load_guesser,
        load_questions,
        load_buzzer,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--json_guess_output", type=str)
    add_general_params(parser)
    add_guesser_params(parser)
    add_buzzer_params(parser)
    add_question_params(parser)

    flags = parser.parse_args()

    setup_logging(flags)

    guesser = load_guesser(flags)
    buzzer = load_buzzer(flags)
    questions = load_questions(flags)

    buzzer.add_data(questions)
    buzzer.build_features(flags.buzzer_history_length, flags.buzzer_history_depth)

    vocab = buzzer.write_json(flags.json_guess_output)
    with open("data/small_guess.vocab", "w") as outfile:
        for ii in vocab:
            outfile.write("%s\n" % ii)
