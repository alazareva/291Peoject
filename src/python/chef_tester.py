import pickle
import numpy as np
import tensorflow as tf
import ConfigParser
import time

from caption_generator import Caption_Generator

def test():

    #Getting all the parameters from configuration file
    config_file_path = 'config.ini'
    hyperparam = 'Hyperparams'
    file_paths = 'Files'
    
    config = ConfigParser.ConfigParser()
    config.read(config_file_path)
    
    test_model_path = config.get(file_paths, 'test_model_path')
    word_to_index_path = config.get(file_paths, 'word_to_index_path')
    index_to_word_path = config.get(file_paths, 'index_to_word_path')
    length_of_longest_sentence_path = config.get(file_paths, 'length_of_longest_sentence_path')
    test_feat = config.get(file_paths, 'test_image_features_path') 

    batch_size = int(config.get(hyperparam, 'batch_size'))
    dim_word_embedding = int(config.get(hyperparam, 'dim_word_embedding'))
    dim_context = int(config.get(hyperparam, 'dim_context'))
    dim_hidden = int(config.get(hyperparam, 'dim_hidden'))
    context_shape_start = int(config.get(hyperparam, 'context_shape_start'))
    context_shape_end = int(config.get(hyperparam, 'context_shape_end'))

    context_shape = [context_shape_start, context_shape_end]
    
    #Get the vocabulary information
    word_to_index_list = pickle.load(open(word_to_index_path,"rb"))
    index_to_word_list = pickle.load(open(index_to_word_path,"rb"))

    n_words = len(word_to_index_list)
    test_image_features = pickle.load(open(test_feat,"rb"))
    length_of_longest_sentence = pickle.load(open(length_of_longest_sentence_path, "rb"))

    session = tf.Session()

    generated_recipes = {}

    caption_generator = Caption_Generator(
            n_words=n_words,
            dim_word_embedding=dim_word_embedding,
            dim_context=dim_context,
            dim_hidden=dim_hidden,
            n_lstm_steps=length_of_longest_sentence,
            batch_size=batch_size,
            context_shape=context_shape)

    context, generated_words, logit_list, alpha_list = caption_generator.build_generator(maxlen=length_of_longest_sentence)
    saver = tf.train.Saver()
    saver.restore(session, test_model_path)
    print("Model Restored")
    
    for rec in test_image_features.keys():
        print("processing",rec)
        test_image_feature = np.array(test_image_features[rec])
        test_image_feature = test_image_feature.reshape(-1, context_shape[1], context_shape[0]).swapaxes(1,2)
        generated_word_index = session.run(generated_words, feed_dict={context:test_image_feature})
        words = [index_to_word_list[x[0]] for x in generated_word_index]
        generated_recipes[rec] = ' '.join(words)

    return generated_recipes

if __name__ == "__main__":

    start = time.time()
    generated_recipes = test()
    print "Total Time taken for testing: %s"%(time.time() - start)