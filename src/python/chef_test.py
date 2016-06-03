import pickle
import numpy as np
import tensorflow as tf
import ConfigParser
import time

from caption_generator import Caption_Generator
from evaluation_metrics import Evaluation_Metric

def run_test(suffix):

    #Getting all the parameters from configuration file
    config_file_path = 'config.ini'
    hyperparam = 'Hyperparams'
    file_paths = 'Files'

    val_metric = Evaluation_Metric()
    
    config = ConfigParser.ConfigParser()
    config.read(config_file_path)
    
    word_to_index_path = config.get(file_paths, 'word_to_index_path')
    index_to_word_path = config.get(file_paths, 'index_to_word_path')
    length_of_longest_sentence_path = config.get(file_paths, 'length_of_longest_sentence_path')
    val_feat = config.get(file_paths, 'val_image_features_path')
    val_recipe_path = config.get(file_paths,'val_recipes_path')
    
    batch_size = int(config.get(hyperparam, 'batch_size'))
    dim_word_embedding = int(config.get(hyperparam, 'dim_word_embedding'))
    dim_context = int(config.get(hyperparam, 'dim_context'))
    dim_hidden = int(config.get(hyperparam, 'dim_hidden'))
    context_shape_start = int(config.get(hyperparam, 'context_shape_start'))
    context_shape_end = int(config.get(hyperparam, 'context_shape_end'))
    model_path = config.get(file_paths,'model_path')
    model_path = model_path + "model-%s"%suffix
    predicted_recipes_path = "predicted_recipes%s"%suffix

    context_shape = [context_shape_start, context_shape_end]
    
    #Get the vocabulary information
    word_to_index_list = pickle.load(open(word_to_index_path,"rb"))
    index_to_word_list = pickle.load(open(index_to_word_path,"rb"))

    n_words = len(word_to_index_list)
    val_image_features = pickle.load(open(val_feat,"rb"))
    length_of_longest_sentence = pickle.load(open(length_of_longest_sentence_path, "rb"))

    session = tf.Session()

    generated_recipes = {}

    caption_generator = Caption_Generator(
            n_words=n_words,
            dim_embed=dim_word_embedding,
            dim_ctx=dim_context,
            dim_hidden=dim_hidden,
            n_lstm_steps=length_of_longest_sentence,
            batch_size=batch_size,
            ctx_shape=context_shape)

    context, generated_words, logit_list, alpha_list = caption_generator.build_generator(maxlen=length_of_longest_sentence)
    saver = tf.train.Saver()
    saver.restore(session, model_path)
    print("Model Restored-%s"%model_path)
    
    for rec in val_image_features.keys():
        val_image_feature = np.array(val_image_features[rec])
        val_image_feature = val_image_feature.reshape(-1, context_shape[1], context_shape[0]).swapaxes(1,2)
        generated_word_index = session.run(generated_words, feed_dict={context:val_image_feature})
        words = [index_to_word_list[x[0]] for x in generated_word_index]
        generated_recipes[rec] = ' '.join(words)
             
    val_recipes = pickle.load(open(val_recipe_path, "rb"))
    metric = val_metric.evaluate(val_recipes,generated_recipes)
    print "Current Velication Metric: ", metric
    pickle.dump(metric,open("metric-%s.p"%suffix, "wb"))

    print "Saving generated recipes .."
    pickle.dump(generated_recipes, open(predicted_recipes_path,"wb"))

    return generated_recipes

if __name__ == "__main__":

    start = time.time()
    model_number = 5
    generated_recipes = run_test(model_number)

    print "Total Time taken for validating %s models: %s"%(model_number, (time.time() - start))