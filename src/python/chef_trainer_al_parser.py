import os
import tensorflow as tf
import numpy as np
import pickle
import time 


from keras.preprocessing import sequence
from caption_generator import Caption_Generator
from vocabulary_builder import build_vocabulary
import ConfigParser
from nltk.tokenize import WordPunctTokenizer

def get_init_bias_vector(word_counts, index_to_word_list):

    init_bias_vector = np.array([1.0*word_counts[index_to_word_list[i]] for i in index_to_word_list])
    init_bias_vector /= np.sum(init_bias_vector) # normalize to frequencies
    init_bias_vector = np.log(init_bias_vector)
    init_bias_vector -= np.max(init_bias_vector) # shift to nice numeric range

    return init_bias_vector

def train(annotation_data, image_features,word_to_index_list, index_to_word_list, bias_init_vector,number_of_epochs,
    batch_size,dim_word_embedding,dim_context,dim_hidden,context_shape_start, context_shape_end, learning_rate,pretrained_model_path=None):
    
    context_shape = [context_shape_start, context_shape_end]
    captions = annotation_data.values()
    number_of_words = len(word_to_index_list)
    length_of_longest_sentence = np.max(map(lambda x: len(tokenizer.tokenize(x), captions))
    print 'Length of the longest sentence is %s' % length_of_longest_sentence

    tokenizer = WordPunctTokenizer()

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
    
    saver = tf.train.Saver(max_to_keep=None)
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
            #print "Start %s End %s"%(start,end)

            current_features = np.array([image_features[x] for x in image_ids[start:end]])
            current_features = current_features.reshape(-1, context_shape[1], context_shape[0]).swapaxes(1, 2)

            current_captions = captions[start:end]
            current_caption_ind = map(
                lambda cap: [word_to_index_list[word] for word in tokenize_recipe(cap, tokenizer) if word in word_to_index_list],
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

        saver.save(session, os.path.join(model_path, 'model'), global_step=epoch)
        print "Time taken for epoch %s is %s"%(epoch,(time.time()-epoch_start_time))


def reduce_dataset_to_size(images, captions, size):
    if size ==-1:
        return images, captions

    images_new = dict(images.items()[:size])

    keys = images_new.keys()
    captions_new = dict()
    
    for key in keys:
        captions_new[key] = captions[key]

    return images_new, captions_new

def trim_sentence_length(captions, trim_to_size):
    tokenizer = WordPunctTokenizer()
    new_dict = dict()
    length_of_longest_sentence = np.max(map(lambda x: len(x.split(' ')), captions.values()))

    if length_of_longest_sentence <= trim_to_size:
        return captions

    for key in captions.keys():
        recipe = captions[key]
        words = tokenize_recipe(recipe, tokenizer)
        list_of_words = words[:trim_to_size]
        new_dict[key]  = ' '.join(map(unicode, list_of_words))
        #sentence_lenths.append(len(new_dict[key]))

    #print np.max(sentence_lenths)

    return new_dict

def tokenize_recipe(recipe, tokenizer):
    return sum([tokenizer.tokenize(line)+['\n'] for line in recipe.lower().split('\n') if len(line)>0], [])

if __name__ == "__main__":

    start = time.time()

    config_file_path = 'config.ini'
    hyperparam = 'Hyperparams'
    file_paths = 'Files'
    
    config = ConfigParser.ConfigParser()
    config.read(config_file_path)
    image_features_path = config.get(file_paths, 'image_features_path')
    model_path = config.get(file_paths, 'model_path')
    word_to_index_path = config.get(file_paths, 'word_to_index_path')
    index_to_word_path = config.get(file_paths, 'index_to_word_path')
    word_count_path = config.get(file_paths, 'word_count_path')
    annotation_path = config.get(file_paths, 'annotation_path')

    number_of_epochs = int(config.get(hyperparam, 'number_of_epochs'))
    batch_size = int(config.get(hyperparam, 'batch_size'))
    dim_word_embedding = int(config.get(hyperparam, 'dim_word_embedding'))
    dim_context = int(config.get(hyperparam, 'dim_context'))
    dim_hidden = int(config.get(hyperparam, 'dim_hidden'))
    context_shape_start = int(config.get(hyperparam, 'context_shape_start'))
    context_shape_end = int(config.get(hyperparam, 'context_shape_end'))
    learning_rate = float(config.get(hyperparam, 'learning_rate'))

    
    #word_to_index_list = pickle.load(open(word_to_index_path), 'rb')
    #index_to_word_list = pickle.load(open(index_to_word_path), 'rb')
    #word_counts = pickle.load(open(word_counts_path), 'rb')    

    annotation_data_raw = pickle.load(open(annotation_path, "rb"))
    annotation_data = trim_sentence_length(annotation_data_raw, trim_to_size =300)

    image_features = np.load(image_features_path)
    images_new, captions_new = reduce_dataset_to_size(image_features, annotation_data, size=100)

    word_to_index_list, index_to_word_list, word_counts = build_vocabulary(captions_new.values(), save_variables=True)
    bias_init_vector = get_init_bias_vector(word_counts, index_to_word_list)
    
    train(annotation_data,image_features, word_to_index_list, index_to_word_list, bias_init_vector,number_of_epochs,
    batch_size,dim_word_embedding,dim_context,dim_hidden,context_shape_start, context_shape_end, learning_rate)
    print "Total Time taken for training: %s"%(time.time() - start)
