import os
import tensorflow as tf
import numpy as np
import pickle

from tensorflow.models.rnn import rnn_cell
import tensorflow.python.platform
from keras.preprocessing import sequence
from caption_generator import Caption_Generator

def get_init_bias_vector(word_counts, index_to_word_list):

    init_bias_vector = np.array([1.0*word_counts[index_to_word_list[i]] for i in index_to_word_list])
    init_bias_vector /= np.sum(init_bias_vector) # normalize to frequencies
    init_bias_vector = np.log(init_bias_vector)
    init_bias_vector -= np.max(init_bias_vector) # shift to nice numeric range

    return init_bias_vector

def get_all_lists():

    word_to_index_path = 'test_data/caption/caption.p'
    word_to_index_list = pickle.load(open(word_to_index_path, "rb"))
    index_to_word_path = ''
    index_to_word_list = pickle.load(open(index_to_word_path, "rb"))
    word_count_path = ''
    word_counts = pickle.load(open(word_count_path, "rb"))
    bias_init_vector = get_init_bias_vector(word_counts, index_to_word_list)

    return word_to_index_list, index_to_word_list, bias_init_vector

def train(pretrained_model_path=None):

    # TODO Move these parameters to a config file
    number_of_epochs = 20
    batch_size = 10
    dim_word_embedding = 256
    dim_context = 2048
    dim_hidden = 256
    #TODO Rename context to something specific
    context_shape = [1, 2048]
    learning_rate = 0.001

    image_features_path = 'test_data/features/features.p'
    model_path = 'test_data/model/'

    word_to_index_list, index_to_word_list, bias_init_vector = get_all_lists()

    #TODO Refactor the below code
    annotation_path = 'test_data/caption/caption.p'
    annotation_data = pickle.load(open(annotation_path, "rb"))
    captions = annotation_data.values()

    number_of_words = len(word_to_index_list)
    image_features = np.load(image_features_path)
    length_of_longest_sentence = np.max(map(lambda x: len(x.split(' ')), captions))

    session = tf.Session()

    caption_generator = Caption_Generator(
            n_words=number_of_words,
            dim_embed=dim_word_embedding,
            dim_ctx=dim_context,
            dim_hidden=dim_hidden,
            n_lstm_steps=length_of_longest_sentence + 1,
            batch_size=batch_size,
            ctx_shape=context_shape,
            bias_init_vector=bias_init_vector)

    loss, context, sentence, mask = caption_generator.build_model()
    saver = tf.train.Saver(max_to_keep=50)

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    tf.initialize_all_variables().run()

    if pretrained_model_path is not None:
        print "Starting with pretrained model"
        saver.restore(session, pretrained_model_path)

    image_ids = annotation_data.keys()

    for epoch in range(number_of_epochs):
        for start, end in zip( \
                range(0, len(captions), batch_size),
                range(batch_size, len(captions), batch_size)):

            current_features = np.array([image_features[x] for x in image_ids[start:end]])
            current_features = current_features.reshape(-1, context_shape[1], context_shape[0]).swapaxes(1, 2)

            current_captions = captions[start:end]
            current_caption_ind = map(
                lambda cap: [word_to_index_list[word] for word in cap.lower().split(' ') if word in word_to_index_list],
                current_captions)  # '.'은 제거

            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=length_of_longest_sentence + 1)

            current_mask_matrix = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array(map(lambda x: (x != 0).sum() + 1, current_caption_matrix))

            for ind, row in enumerate(current_mask_matrix):
                row[:nonzeros[ind]] = 1

            _, loss_value = session.run([train_op, loss], feed_dict={
                context: current_features,
                sentence: current_caption_matrix,
                mask: current_mask_matrix})

            print "Current Cost: ", loss_value
        saver.save(session, os.path.join(model_path, 'model'), global_step=epoch)


if __name__ == "__main__":
    train()
