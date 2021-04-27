#!/usr/bin/env python
# tensorflow v2.4.0
# keras v2.4.3
from keras.layers import *
from keras import Model
from keras.models import load_model
from keras import optimizers


def predict_model(learning_rate=0.001, drop_rate=0.5):
    # ttop: the type of padding
    ttop = 'same'
    S = 1
    I = 400
    length_bases = 4
    feature_num = 163

    ###       ###
    ### left ###
    ###       ###
    # layer 1 ~ 2
    left_input_layer = Input(shape=(S+I, length_bases), name="left_input")
    left_conv1d_1 = Conv1D(32, 1,  padding=ttop, dilation_rate=1, name='left_conv1d_1')(left_input_layer)

    # layer 3 ~ 9
    left_batch_normalization_1 = BatchNormalization(name='left_batch_normalization_1')(left_conv1d_1)
    left_activation_1 = Activation(activation='relu', name='left_activation_1')(left_batch_normalization_1)
    left_conv1d_3 = Conv1D(32, 11, padding=ttop, dilation_rate=1,name='left_conv1d_3')(left_activation_1)
    left_batch_normalization_2 = BatchNormalization(name='left_batch_normalization_2')(left_conv1d_3)
    left_activation_2 = Activation(activation='relu', name='left_activation_2')(left_batch_normalization_2)
    left_conv1d_4 = Conv1D(32, 11, padding=ttop, dilation_rate=1, name='left_conv1d_4')(left_activation_2)
    left_add_1 = Add(name='left_add_1')([left_conv1d_4, left_conv1d_1])

    # layer 10 ~ 16
    left_batch_normalization_3 = BatchNormalization(name='left_batch_normalization_3')(left_add_1)
    left_activation_3 = Activation(activation='relu', name='left_activation_3')(left_batch_normalization_3)
    left_conv1d_5 = Conv1D(32, 11, padding=ttop, dilation_rate=1, name='left_conv1d_5')(left_activation_3)
    left_batch_normalization_4 = BatchNormalization(name='left_batch_normalization_4')(left_conv1d_5)
    left_activation_4 = Activation(activation='relu', name='left_activation_4')(left_batch_normalization_4)
    left_conv1d_6 = Conv1D(32, 11, padding=ttop, dilation_rate=1, name='left_conv1d_6')(left_activation_4)
    left_add_2 = Add(name='left_add_2')([left_conv1d_6, left_add_1])

    # layer 17 ~ 23
    left_batch_normalization_5 = BatchNormalization(name='left_batch_normalization_5')(left_add_2)
    left_activation_5 = Activation(activation='relu', name='left_activation_5')(left_batch_normalization_5)
    left_conv1d_7 = Conv1D(32, 11, padding=ttop, dilation_rate=1, name='left_conv1d_7')(left_activation_5)
    left_batch_normalization_6 = BatchNormalization(name='left_batch_normalization_6')(left_conv1d_7)
    left_activation_6 = Activation(activation='relu', name='left_activation_6')(left_batch_normalization_6)
    left_conv1d_8 = Conv1D(32, 11, padding=ttop, dilation_rate=1, name='left_conv1d_8')(left_activation_6)
    left_add_3 = Add(name='left_add_3')([left_conv1d_8, left_add_2])

    # layer 24 ~ 30
    left_batch_normalization_7 = BatchNormalization(name='left_batch_normalization_7')(left_add_3)
    left_activation_7 = Activation(activation='relu', name='left_activation_7')(left_batch_normalization_7)
    left_conv1d_9 = Conv1D(32, 11, padding=ttop, dilation_rate=1, name='left_conv1d_9')(left_activation_7)
    left_batch_normalization_8 = BatchNormalization(name='left_batch_normalization_8')(left_conv1d_9)
    left_activation_8 = Activation(activation='relu', name='left_activation_8')(left_batch_normalization_8)
    left_conv1d_10 = Conv1D(32, 11, padding=ttop, dilation_rate=1, name='left_conv1d_10')(left_activation_8)
    left_add_4 = Add(name='left_add_4')([left_conv1d_10, left_add_3])

    # layer 31 ~ 37
    left_batch_normalization_9 = BatchNormalization(name='left_batch_normalization_9')(left_add_4)
    left_activation_9 = Activation(activation='relu', name='left_activation_9')(left_batch_normalization_9)
    left_conv1d_12 = Conv1D(32, 11, padding=ttop, dilation_rate=4, name='left_conv1d_12')(left_activation_9)
    left_batch_normalization_10 = BatchNormalization(name='left_batch_normalization_10')(left_conv1d_12)
    left_activation_10 = Activation(activation='relu', name='left_activation_10')(left_batch_normalization_10)
    left_conv1d_13 = Conv1D(32, 11, padding=ttop, dilation_rate=4, name='left_conv1d_13')(left_activation_10)
    left_add_6 = Add(name='left_add_6')([left_conv1d_13, left_add_4])

    # layer 38 ~ 44
    left_batch_normalization_11 = BatchNormalization(name='left_batch_normalization_11')(left_add_6)
    left_activation_11 = Activation(activation='relu', name='left_activation_11')(left_batch_normalization_11)
    left_conv1d_14 = Conv1D(32, 11, padding=ttop, dilation_rate=4, name='left_conv1d_14')(left_activation_11)
    left_batch_normalization_12 = BatchNormalization(name='left_batch_normalization_12')(left_conv1d_14)
    left_activation_12 = Activation(activation='relu', name='left_activation_12')(left_batch_normalization_12)
    left_conv1d_15 = Conv1D(32, 11, padding=ttop, dilation_rate=4, name='left_conv1d_15')(left_activation_12)
    left_add_7 = Add(name='left_add_7')([left_conv1d_15, left_add_6])

    # layer 45 ~ 51
    left_batch_normalization_13 = BatchNormalization(name='left_batch_normalization_13')(left_add_7)
    left_activation_13 = Activation(activation='relu', name='left_activation_13')(left_batch_normalization_13)
    left_conv1d_16 = Conv1D(32, 11, padding=ttop, dilation_rate=4, name='left_conv1d_16')(left_activation_13)
    left_batch_normalization_14 = BatchNormalization(name='left_batch_normalization_14')(left_conv1d_16)
    left_activation_14 = Activation(activation='relu', name='left_activation_14')(left_batch_normalization_14)
    left_conv1d_17 = Conv1D(32, 11, padding=ttop, dilation_rate=4, name='left_conv1d_17')(left_activation_14)
    left_add_8 = Add(name='left_add_8')([left_conv1d_17, left_add_7])

    # layer 52 ~ 57
    left_batch_normalization_15 = BatchNormalization(name='left_batch_normalization_15')(left_add_8)
    left_activation_15 = Activation(activation='relu', name='left_activation_15')(left_batch_normalization_15)
    left_conv1d_18 = Conv1D(32, 11, padding=ttop, dilation_rate=4, name='left_conv1d_18')(left_activation_15)
    left_batch_normalization_16 = BatchNormalization(name='left_batch_normalization_16')(left_conv1d_18)
    left_activation_16 = Activation(activation='relu', name='left_activation_16')(left_batch_normalization_16)
    left_conv1d_19 = Conv1D(32, 11, padding=ttop, dilation_rate=4, name='left_conv1d_19')(left_activation_16)

    # layer 58 ~ 62
    left_conv1d_2 = Conv1D(32, 1, padding=ttop, dilation_rate=1, name='left_conv1d_2')(left_conv1d_1)
    left_conv1d_11 = Conv1D(32, 1, padding=ttop, dilation_rate=1, name='left_conv1d_11')(left_add_4)
    left_add_9 = Add(name='left_add_9')([left_conv1d_19, left_add_8])
    left_add_5 = Add(name='left_add_5')([left_conv1d_2, left_conv1d_11])
    left_conv1d_20 = Conv1D(32, 1, padding=ttop, dilation_rate=1, name='left_conv1d_20')(left_add_9)

    # layer 63 ~ 64
    left_add_10 = Add(name='left_add_10')([left_add_5, left_conv1d_20])
    left_cropping1d = Cropping1D(cropping=(int(S/2)), name='left_cropping1d')(left_add_10)

    ###       ###
    ### right ###
    ###       ###
    # layer 1 ~ 2
    right_input_layer = Input(shape=(S+I, length_bases), name="right_input")
    right_conv1d_1 = Conv1D(32, 1,  padding=ttop, dilation_rate=1, name='right_conv1d_1')(right_input_layer)

    # layer 3 ~ 9
    right_batch_normalization_1 = BatchNormalization(name='right_batch_normalization_1')(right_conv1d_1)
    right_activation_1 = Activation(activation='relu', name='right_activation_1')(right_batch_normalization_1)
    right_conv1d_3 = Conv1D(32, 11, padding=ttop, dilation_rate=1, name='right_conv1d_3')(right_activation_1)
    right_batch_normalization_2 = BatchNormalization(name='right_batch_normalization_2')(right_conv1d_3)
    right_activation_2 = Activation(activation='relu', name='right_activation_2')(right_batch_normalization_2)
    right_conv1d_4 = Conv1D(32, 11, padding=ttop, dilation_rate=1, name='right_conv1d_4')(right_activation_2)
    right_add_1 = Add(name='right_add_1')([right_conv1d_4, right_conv1d_1])

    # layer 10 ~ 16
    right_batch_normalization_3 = BatchNormalization(name='right_batch_normalization_3')(right_add_1)
    right_activation_3 = Activation(activation='relu', name='right_activation_3')(right_batch_normalization_3)
    right_conv1d_5 = Conv1D(32, 11, padding=ttop, dilation_rate=1, name='right_conv1d_5')(right_activation_3)
    right_batch_normalization_4 = BatchNormalization(name='right_batch_normalization_4')(right_conv1d_5)
    right_activation_4 = Activation(activation='relu', name='right_activation_4')(right_batch_normalization_4)
    right_conv1d_6 = Conv1D(32, 11, padding=ttop, dilation_rate=1, name='right_conv1d_6')(right_activation_4)
    right_add_2 = Add(name='right_add_2')([right_conv1d_6, right_add_1])

    # layer 17 ~ 23
    right_batch_normalization_5 = BatchNormalization(name='right_batch_normalization_5')(right_add_2)
    right_activation_5 = Activation(activation='relu', name='right_activation_5')(right_batch_normalization_5)
    right_conv1d_7 = Conv1D(32, 11, padding=ttop, dilation_rate=1, name='right_conv1d_7')(right_activation_5)
    right_batch_normalization_6 = BatchNormalization(name='right_batch_normalization_6')(right_conv1d_7)
    right_activation_6 = Activation(activation='relu', name='right_activation_6')(right_batch_normalization_6)
    right_conv1d_8 = Conv1D(32, 11, padding=ttop, dilation_rate=1, name='right_conv1d_8')(right_activation_6)
    right_add_3 = Add(name='right_add_3')([right_conv1d_8, right_add_2])

    # layer 24 ~ 30
    right_batch_normalization_7 = BatchNormalization(name='right_batch_normalization_7')(right_add_3)
    right_activation_7 = Activation(activation='relu', name='right_activation_7')(right_batch_normalization_7)
    right_conv1d_9 = Conv1D(32, 11, padding=ttop, dilation_rate=1, name='right_conv1d_9')(right_activation_7)
    right_batch_normalization_8 = BatchNormalization(name='right_batch_normalization_8')(right_conv1d_9)
    right_activation_8 = Activation(activation='relu', name='right_activation_8')(right_batch_normalization_8)
    right_conv1d_10 = Conv1D(32, 11, padding=ttop, dilation_rate=1, name='right_conv1d_10')(right_activation_8)
    right_add_4 = Add(name='right_add_4')([right_conv1d_10, right_add_3])

    # layer 31 ~ 37
    right_batch_normalization_9 = BatchNormalization(name='right_batch_normalization_9')(right_add_4)
    right_activation_9 = Activation(activation='relu', name='right_activation_9')(right_batch_normalization_9)
    right_conv1d_12 = Conv1D(32, 11, padding=ttop, dilation_rate=4, name='right_conv1d_12')(right_activation_9)
    right_batch_normalization_10 = BatchNormalization(name='right_batch_normalization_10')(right_conv1d_12)
    right_activation_10 = Activation(activation='relu', name='right_activation_10')(right_batch_normalization_10)
    right_conv1d_13 = Conv1D(32, 11, padding=ttop, dilation_rate=4, name='right_conv1d_13')(right_activation_10)
    right_add_6 = Add(name='right_add_6')([right_conv1d_13, right_add_4])

    # layer 38 ~ 44
    right_batch_normalization_11 = BatchNormalization(name='right_batch_normalization_11')(right_add_6)
    right_activation_11 = Activation(activation='relu', name='right_activation_11')(right_batch_normalization_11)
    right_conv1d_14 = Conv1D(32, 11, padding=ttop, dilation_rate=4, name='right_conv1d_14')(right_activation_11)
    right_batch_normalization_12 = BatchNormalization(name='right_batch_normalization_12')(right_conv1d_14)
    right_activation_12 = Activation(activation='relu', name='right_activation_12')(right_batch_normalization_12)
    right_conv1d_15 = Conv1D(32, 11, padding=ttop, dilation_rate=4, name='right_conv1d_15')(right_activation_12)
    right_add_7 = Add(name='right_add_7')([right_conv1d_15, right_add_6])

    # layer 45 ~ 51
    right_batch_normalization_13 = BatchNormalization(name='right_batch_normalization_13')(right_add_7)
    right_activation_13 = Activation(activation='relu', name='right_activation_13')(right_batch_normalization_13)
    right_conv1d_16 = Conv1D(32, 11, padding=ttop, dilation_rate=4, name='right_conv1d_16')(right_activation_13)
    right_batch_normalization_14 = BatchNormalization(name='right_batch_normalization_14')(right_conv1d_16)
    right_activation_14 = Activation(activation='relu', name='right_activation_14')(right_batch_normalization_14)
    right_conv1d_17 = Conv1D(32, 11, padding=ttop, dilation_rate=4, name='right_conv1d_17')(right_activation_14)
    right_add_8 = Add(name='right_add_8')([right_conv1d_17, right_add_7])

    # layer 52 ~ 57
    right_batch_normalization_15 = BatchNormalization(name='right_batch_normalization_15')(right_add_8)
    right_activation_15 = Activation(activation='relu', name='right_activation_15')(right_batch_normalization_15)
    right_conv1d_18 = Conv1D(32, 11, padding=ttop, dilation_rate=4, name='right_conv1d_18')(right_activation_15)
    right_batch_normalization_16 = BatchNormalization(name='right_batch_normalization_16')(right_conv1d_18)
    right_activation_16 = Activation(activation='relu', name='right_activation_16')(right_batch_normalization_16)
    right_conv1d_19 = Conv1D(32, 11, padding=ttop, dilation_rate=4, name='right_conv1d_19')(right_activation_16)

    # layer 58 ~ 62
    right_conv1d_2 = Conv1D(32, 1, padding=ttop, dilation_rate=1, name='right_conv1d_2')(right_conv1d_1)
    right_conv1d_11 = Conv1D(32, 1, padding=ttop, dilation_rate=1, name='right_conv1d_11')(right_add_4)
    right_add_9 = Add(name='right_add_9')([right_conv1d_19, right_add_8])
    right_add_5 = Add(name='right_add_5')([right_conv1d_2, right_conv1d_11])
    right_conv1d_20 = Conv1D(32, 1, padding=ttop, dilation_rate=1, name='right_conv1d_20')(right_add_9)

    # layer 63 ~ 64
    right_add_10 = Add(name='right_add_10')([right_add_5, right_conv1d_20])
    right_cropping1d = Cropping1D(cropping=(int(S/2)), name='right_cropping1d')(right_add_10)

    ###       ###
    ### merge ###
    ###       ###
    left_flatten = Flatten(name='left_flatten')(left_cropping1d)
    right_flatten = Flatten(name='right_flatten')(right_cropping1d)
    feature_input_layer = Input(shape=(feature_num, ), name="feature_input")
    concat_left_right_feature = Concatenate(name='concat_left_right_feature')([left_flatten, right_flatten, feature_input_layer])

    dense_1 = Dense(2048, name='dense_1')(concat_left_right_feature)
    batch_normalization_dense_1 = BatchNormalization(name='batch_normalization_dense_1')(dense_1)
    drop_dense_1 = Dropout(drop_rate, name='drop_dense_1')(batch_normalization_dense_1)
    activation_dense_1 = Activation(activation='relu', name='activation_dense_1')(drop_dense_1)
    dense_2 = Dense(512, name='dense_2')(activation_dense_1)
    batch_normalization_dense_2 = BatchNormalization(name='batch_normalization_dense_2')(dense_2)
    drop_dense_2 = Dropout(drop_rate, name='drop_dense_2')(batch_normalization_dense_2)
    activation_dense_2 = Activation(activation='relu', name='activation_dense_2')(drop_dense_2)
    dense_3 = Dense(64, name='dense_3')(activation_dense_2)
    batch_normalization_dense_3 = BatchNormalization(name='batch_normalization_dense_3')(dense_3)
    drop_dense_3 = Dropout(drop_rate, name='drop_dense_3')(batch_normalization_dense_3)
    activation_dense_3 = Activation(activation='relu', name='activation_dense_3')(drop_dense_3)

    output_layer = Dense(1, activation='sigmoid', name='output_layer')(activation_dense_3)
    model = Model(inputs=[left_input_layer, right_input_layer, feature_input_layer], outputs=output_layer)

    model.compile(loss="binary_crossentropy",
                      optimizer=optimizers.Adam(
                              lr=learning_rate,
                              beta_1=0.9,
                              beta_2=0.999,
                              epsilon=None,
                              decay=0.0),
                      metrics=['accuracy'])
    return model
