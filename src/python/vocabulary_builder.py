import numpy as np
import pickle

def build_vocabulary(sentence_iterator, word_count_threshold=0): # borrowed this function from NeuralTalk

    print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, )
    word_counts = {}
    number_of_sentences = 0
    for sentence in sentence_iterator:
        number_of_sentences += 1
        for current_word in sentence.lower().split(' '):
            word_counts[current_word] = word_counts.get(current_word, 0) + 1
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

    word_counts['#END#'] = number_of_sentences

    #TODO Save the information

    return word_to_index_list, index_to_word_list



if __name__ == "__main__":

    annotation_path = 'test_data/caption/caption.p'
    annotation_data = pickle.load(open(annotation_path, "rb"))
    captions = annotation_data.values()
    build_vocabulary(captions)