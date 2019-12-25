# =============================================================================
# Import Libraries
# =============================================================================
import os
import json
from segnet3d import create_model
from datagen import SyntheticSeisGen, DataGenerator
from time import time
from metrics import cross_entropy_balanced
from keras import callbacks, optimizers
from keras.models import model_from_json

def train_model(Dataset,CompileModel,TrainModel,name_data,name_model,name_weights,
             num_data_tr,num_data_val,patch_size,num_epochs,lr,params):
# =============================================================================
# Configuration
# =============================================================================
    ''' Paths for dataset '''
    path_home = os.getcwd()
    tr_path = os.path.join(path_home, 'dataset', name_data, 'train')
    vl_path = os.path.join(path_home, 'dataset', name_data, 'validation')
    tdpath = os.path.join(tr_path, 'seis')
    tfpath = os.path.join(tr_path, 'fault')
    vdpath = os.path.join(vl_path, 'seis')
    vfpath = os.path.join(vl_path, 'fault')

    ''' Paths for model, weights, and metircs '''    
    path_model = os.path.join(path_home, 'model')
    path_model_arch = os.path.join(path_model, name_model + '.json')
    path_weights = os.path.join(path_model, 'weights', name_weights + '.h5')
    path_hists = os.path.join(path_model, 'weights', name_weights + '_hist.txt')
    path_cb = os.path.join(path_model, 'call_back', name_weights)

    t0 = time()
# =============================================================================
# Build Training & Validation Dataset
# =============================================================================
    if Dataset:
        ''' Generate Synthetic Data '''
        print('Generating Training Data')
        SyntheticSeisGen(tr_path, num_data_tr, patch_size)
        print('\nGenerating Validation Data')
        SyntheticSeisGen(vl_path, num_data_val, patch_size)
        print('\nSaving Dataset')
    else:
        if not (os.path.exists(tr_path) | os.path.exists(vl_path)):
            print("Please Create Dataset First!")

# =============================================================================
# Create 3D Convolutional Neural Network Model
# =============================================================================
    if CompileModel:
        ''' Compile CNN Model '''
        print('Creating CNN Model')
        conv_model = create_model((*[int(patch_size)]*3,1), lr)
        model = conv_model.model
        ''' Save CNN Model '''
        json_string = model.to_json()
        open(path_model_arch,'w').write(json_string)        
    else:
        ''' Load CNN Model '''
        print('Loading CNN Model')

        json_file = open(path_model_arch, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)    
        model.compile(optimizer=optimizers.Adam(lr),
                      loss=cross_entropy_balanced, metrics=['accuracy'])

# =============================================================================
# Train CNN Model    
# =============================================================================
    if TrainModel:
        ''' Callbacks Configuration '''
        cp_fn = os.path.join(path_cb, 'checkpoint.{epoch:02d}.h5')
        cp_cb = callbacks.ModelCheckpoint(filepath=cp_fn, verbose=1, save_best_only=False)
        csv_fn = os.path.join(path_cb, 'train_log.csv')
        csv_cb = callbacks.CSVLogger(csv_fn, append=True, separator=';')
        tb_cb = callbacks.TensorBoard(log_dir=path_cb, histogram_freq=0, batch_size=2,
                                      write_graph=True, write_grads=True, write_images=True)
        cbks = [cp_cb, csv_cb, tb_cb]
           
        ''' Train CNN Model '''
        print('\nModel Fitting')
        tdata_IDs = range(num_data_tr)
        vdata_IDs = range(num_data_val)
        tr_gen = DataGenerator(dpath=tdpath,fpath=tfpath,data_IDs=tdata_IDs,**params)
        val_gen = DataGenerator(dpath=vdpath,fpath=vfpath,data_IDs=vdata_IDs,**params)
        history = model.fit_generator(generator=tr_gen, validation_data=val_gen,
                                      epochs=num_epochs,verbose=1,callbacks=cbks)
        history_dict = history.history

        ''' Save Weights & Metrics '''
        model.save_weights(path_weights)
        json.dump(history_dict, open(path_hists, 'w'))        
    else:
        ''' Load Metrics & Trained Model Weights '''
        model.load_weights(path_weights)
        history_dict = json.load(open(path_hists, 'r'))
        
    print('\nElapsed time: ' + "{:.2f}".format((time()-t0)/60) + ' min')
    return model, history_dict
