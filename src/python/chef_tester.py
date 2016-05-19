import pickle
import numpy as np
import tensorflow as tf

from chef_trainer import get_all_lists
from caption_generator import Caption_Generator

def test(test_feat="test_data/features/test_feat.p", model_path='test_data/model/model-19'):


    batch_size = 10
    dim_word_embedding = 256
    dim_context = 2048
    dim_hidden = 256
    context_shape = [1, 2048]
    length_of_longest_sentence =0

    word_to_index_list, index_to_word_list, bias_init_vector = get_all_lists()

    n_words = len(word_to_index_list)
    feat = np.load(test_feat)
    feat = np.array([feat[x] for x in ['200744']])
    feat = feat.reshape(-1, context_shape[1], context_shape[0]).swapaxes(1,2)

    sess = tf.InteractiveSession()
    #tf.reset_default_graph()

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
    saver.restore(sess, model_path)
    print("Model Restored")

    generated_word_index = sess.run(generated_words, feed_dict={context:feat})
    generated_words = [index_to_word_list[x[0]] for x in generated_word_index]
    return generated_words