import os
import tensorflow as tf
import numpy as np
import pickle
import time 
from tensorflow.models.rnn import rnn_cell
import tensorflow.python.platform
from keras.preprocessing import sequence
from caption_generator import Caption_Generator
from vocabulary_builder import build_vocabulary

image_features_path = '../../pca_train_feat_dict.p'
model_path = '../model/paul/'
word_to_index_path = '../../word_to_index.p'
index_to_word_path = '../../index_to_word.p'
word_count_path = '../../word_count.p'
annotation_path = '../../training_set_recipes.p'

def get_init_bias_vector(word_counts, index_to_word_list):

    init_bias_vector = np.array([1.0*word_counts[index_to_word_list[i]] for i in index_to_word_list])
    init_bias_vector /= np.sum(init_bias_vector) # normalize to frequencies
    init_bias_vector = np.log(init_bias_vector)
    init_bias_vector -= np.max(init_bias_vector) # shift to nice numeric range

    return init_bias_vector

def train(annotation_data, image_features,word_to_index_list, index_to_word_list, bias_init_vector,pretrained_model_path=None):

    # TODO Move these parameters to a config file
    number_of_epochs = 1
    batch_size = 10
    dim_word_embedding = 256
    dim_context = 512
    dim_hidden = 256
    
    #TODO Rename context to something specific
    context_shape = [1, 512]
    learning_rate = 0.01

    captions = annotation_data.values()
    number_of_words = len(word_to_index_list)
    length_of_longest_sentence = np.max(map(lambda x: len(x.split(' ')), captions))
    print 'Length of the longest sentence is %s' % length_of_longest_sentence

    #Go Crazy. Reduce the #sentences to see impact on performance
    length_of_longest_sentence = 99

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

    print "Created caption generator"

    loss, context, sentence, mask = caption_generator.build_model()
    print "Built Model Successfully"
    
    saver = tf.train.Saver(max_to_keep=50)
    print 'Keep your fingers crossed. The training begins!!'
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    tf.initialize_all_variables().run(session=session)

    if pretrained_model_path is not None:
        print "Starting with pretrained model"
        saver.restore(session, pretrained_model_path)

    image_ids = annotation_data.keys()

    print "Running through epochs now"


    for epoch in range(number_of_epochs):
        epoch_start_time = time.time()
        value_x = zip( \
                range(0, len(captions), batch_size),
                range(batch_size, len(captions), batch_size))
        print len(value_x)
        count = 0
        for start, end in zip( \
                range(0, len(captions), batch_size),
                range(batch_size, len(captions), batch_size)):

            count = count + 1
            if count >100:
                break;
            start_iter_time = time.time()
            print "Start %s End %s"%(start,end)

            current_features = np.array([image_features[x] for x in image_ids[start:end]])
            current_features = current_features.reshape(-1, context_shape[1], context_shape[0]).swapaxes(1, 2)

            current_captions = captions[start:end]
            current_caption_ind = map(
                lambda cap: [word_to_index_list[word] for word in cap.lower().split(' ') if word in word_to_index_list],
                current_captions)

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
            print "Time taken %s"%(time.time() - start_iter_time)
        #saver.save(session, os.path.join(model_path, 'model'), global_step=epoch)
        print "Time taken for epoch %s is %s"%(epoch,(time.time()-epoch_start_time))


def reduce_dataset_to_size(images, captions, size):
    images_new = dict(images.items()[:size])

    keys = images_new.keys()
    captions_new = dict()
    
    for key in keys:
        captions_new[key] = captions[key]

    return images_new, captions_new

if __name__ == "__main__":
    
    start = time.time()
    word_to_index_list = pickle.load(open(word_to_index_path), 'rb')
    index_to_word_list = pickle.load(open(index_to_word_path), 'rb')
    word_counts = pickle.load(open(word_counts_path), 'rb')    

    annotation_data = pickle.load(open(annotation_path, "rb"))
    image_features = np.load(image_features_path)
    #images_new, captions_new = reduce_dataset_to_size(image_features, annotation_data,5000)
    #word_to_index_list, index_to_word_list, word_counts = build_vocabulary(captions_new.values())
    bias_init_vector = get_init_bias_vector(word_counts, index_to_word_list)
    
    train(annotation_data,image_features, word_to_index_list, index_to_word_list, bias_init_vector)
    print "Total Time taken for training: %s"%(time.time() - start)
