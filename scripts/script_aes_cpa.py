import aisy_sca
import math
from app import *
from clr import OneCycleLR
from custom.custom_models.neural_networks import *

neurons = 10
learning_rate = 0.0001
starting_filters = 8


def jin_block_pool(input_tensor, num_filters, add_conv, stride=1, kernel_size=11, activation_func='selu', pool=True, avgpool=True):
    y = Conv1D(filters=num_filters, kernel_size=kernel_size, padding='same', strides=stride)(input_tensor)
    y = Activation(activation_func)(y)
    #y = BatchNormalization()(y)
    y = Conv1D(filters=num_filters, kernel_size=kernel_size, padding='same')(y)
    res_connection = input_tensor
    if add_conv:
        res_connection = Conv1D(kernel_size=1, filters=num_filters, strides=stride, padding='same')(input_tensor)
    y = Add()([res_connection, y])
    y = Activation(activation_func)(y)
    #y = BatchNormalization()(y)
    if not pool:
        return y
    return AveragePooling1D(2, 2)(y)


def create_resnet_func(res_block, num_resblocks=10):
    def resnet_newest(classes, number_of_samples):
        activation_func = 'selu'

        input_shape = (number_of_samples, 1)
        inputs = Input(shape=input_shape)
        print(inputs.shape)
        t = res_block(inputs, starting_filters, True, kernel_size=2)
        for j in range(1, num_resblocks):
            print(j, t.shape)
            print(j < (num_resblocks-1))
            t = res_block(t, min(starting_filters * pow(2, j), 256), True, kernel_size=2, pool= j < (num_resblocks -1), avgpool= j > num_resblocks/2)
        t = GlobalAveragePooling1D()(t)
        t = Dense(neurons, activation=activation_func)(t)
        t = Dense(neurons, activation=activation_func)(t)
        t = Dense(9, activation='softmax')(t)
        model = Model(inputs, t, name='resnet')
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    return resnet_newest


aisy = aisy_sca.Aisy()
aisy.set_resources_root_folder(resources_root_folder)
aisy.set_database_root_folder(databases_root_folder)
aisy.set_datasets_root_folder(datasets_root_folder)
aisy.set_database_name("database_ascad.sqlite")
aisy.set_dataset(datasets_dict["aes_hd_mm_ext.h5"])
aisy.set_batch_size(50)
aisy.set_number_of_profiling_traces(200000)
aisy.set_number_of_attack_traces(5000)
aisy.set_epochs(2)
aisy.set_neural_network(create_resnet_func(jin_block_pool, num_resblocks=8))

one_cycle_pol = OneCycleLR(max_lr=1e-3, end_percentage=0.2, scale_percentage=0.1, maximum_momentum=None,
                           minimum_momentum=None, verbose=True)
#aisy.add_callback(one_cycle_pol)

target_byte = 11

early_stopping = {
    "metrics": {
        # "not": {
        #     "direction": "min",
        #     "class": "custom.custom_metrics.number_of_traces",
        #     "parameters": []
        # },
        "loss": {
            "direction": "min",
            "class": "custom.custom_metrics.loss",
            "parameters": []
        }
    }
}
aisy.set_aes_leakage_model(leakage_model="HD",
                           byte=target_byte,
                           direction="Encryption",
                           cipher="AES128",
                           round_first=10,
                           target_state_second="Sbox",
                           target_state_first="Output",
                           round_second=10,
                           attack_direction="output")
key_rank_report_interval = 10
aisy.run(key_rank_report_interval=key_rank_report_interval, early_stopping=early_stopping, key_rank_attack_traces=2000)#, visualization=[4000])
prefix = 'chapter2results/aes_hd_results/identity/'
model_name = 'AES_HD_cnn_id_run{}'.format(1)
# np.save(prefix + model_name + '_ge', aisy)
#
# for i in range(len(aisy.ge_attack_early_stopping)):
#     np.save(prefix + model_name + 'es_ge_' +
#             aisy.ge_attack_early_stopping[i]['metric'],
#             aisy.ge_attack_early_stopping[i]['guessing_entropy'])
#
# metrics_validation = aisy.get_metrics_validation()
# for metric in metrics_validation:
#     np.save(prefix + model_name + metric['metric'],
#             metric['values'])