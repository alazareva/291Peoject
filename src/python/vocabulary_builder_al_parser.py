import numpy as np
import pickle
import ConfigParser
from nltk.tokenize import WordPunctTokenizer

def build_vocabulary(recipe_iterator, save_variables=False,word_count_threshold =0): # borrowed this function from NeuralTalk

    file_paths = 'Files'
    config = ConfigParser.ConfigParser()
    config.read('config.ini')
    word_to_index_path = config.get(file_paths, 'word_to_index_path')
    index_to_word_path = config.get(file_paths, 'index_to_word_path')
    word_count_path = config.get(file_paths, 'word_count_path')

    tokenizer = WordPunctTokenizer()

    print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold )
    length_of_longest_recipe = np.max(map(lambda x: len(tokenizer.tokenize(x)), recipe_iterator))
    print 'Length of the longest sentence is %s'%length_of_longest_sentence
    pickle.dump(length_of_longest_recipe, open('../../config/maxlen.p', "wb"))
    word_counts = {}
    number_of_recipes = 0
    for recipe in recipe_iterator:
        number_of_recipes += 1
        for line in recipe.lower().split('\n'):
            for current_word in tokenizer.tokenize(line):
                word_counts[current_word] = word_counts.get(current_word, 0) + 1
            word_counts['\n']= word_counts.get('\n', 0) + 1
    vocab = [current_word for current_word in word_counts if word_counts[current_word] >= word_count_threshold]
    print 'filtered words from %d to %d' % (len(word_counts), len(vocab))

    index_to_word_list = {}
    index_to_word_list[0] = '#END#'  # end token at the end of the sentence. make first dimension be end token
    word_to_index_list = {}
    word_to_index_list['#START#'] = 0 # make first vector be the start token
    current_index = 1

    for current_word in vocab:
        word_to_index_list[current_word] = current_index
        index_to_word_list[current_index] = current_word
        current_index += 1

    word_counts['#END#'] = number_of_recipes

    if save_variables:
        print 'Completed processing captions. Saving work now ...'
        pickle.dump(word_to_index_list, open(word_to_index_path, "wb"))
        pickle.dump(index_to_word_list, open(index_to_word_path, "wb"))
        pickle.dump(word_counts, open(word_count_path, "wb"))

    return word_to_index_list, index_to_word_list, word_counts