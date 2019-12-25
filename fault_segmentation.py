import os
from utility import plot_metrics, vlm_slicer_synth, vlm_slicer_interactive
from utility import load_data_synth, load_data_field
from training import train_model
from prediction import pred_segy, pred_subvlms

def main():
# =============================================================================
# Create Synthetic seismic cubes and Train U-net based CNN
# =============================================================================
    Dataset = True
    CompileModel = False
    TrainModel = False

    ''' Configuration for Dataset Creation '''
    patch_size = 128
    num_data_tr = 200
    num_data_val = 20

    ''' Model Configuration '''
    name_model = 'SegNet3D'
    lr=1e-4
    num_epochs = 25
    params = {'batch_size': 1, 'num_data_aug': 3, 'dim':([patch_size]*3),
              'n_channels': 1, 'shuffle': True}
    name_weights = 'model_lr_1.0e-04_25_epochs'
#    name_weights = 'model_lr_'+"{:.1e}".format(lr)+'_'+str(num_epochs)+'_epochs'
    name_dataset = '12.06.2019'
    model, history_dict = train_model(Dataset, CompileModel, TrainModel,
                                      name_dataset, name_model,
                                      name_weights, num_data_tr,
                                      num_data_val, patch_size,
                                      num_epochs, lr, params)
    plot_metrics(history_dict)
#    model = load_trained_model(path_home,name_model,name_weights[1])

# =============================================================================
# Apply trained network to field data to predict fault probability
# =============================================================================
    path_home = './'
    path_seis = './F3_seismic.sgy'
    name_subset = ['train','validation']
    size_data = (128*1,128*1,128*1)
    idx0_subvlm = (150,200,330)

    ''' Field Data Application '''
    pred_segy(model, path_home, path_seis, name_dataset[0],'F3',size_data,idx0_subvlm)
    ''' Synthetic Data Application '''
    pred_subvlms(path_home, name_dataset[0], name_subset[0], model) # Training Data
    pred_subvlms(path_home, name_dataset[0], name_subset[1], model) # Validation Data

# =============================================================================
# Display images
# =============================================================================
    name_file_synth = '50.dat'    
    seis_vlm, fault_vlm, pred_vlm = load_data_synth(path_home,
                                                    name_dataset[1],
                                                    name_subset[0],
                                                    name_file_synth,
                                                    (128,128,128))

    vlm_slicer_synth(seis_vlm, fault_vlm, pred_vlm, idx=(63,63,63))
    seis_vlm, pred_vlm = load_data_field(path_home, name_dataset[0], 'F3', size_data)
    vlm_slicer_interactive(seis_vlm, pred_vlm, 2)
    
if __name__ == "__main__":
    main()
