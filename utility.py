import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

''' Display slices '''
def plot_metrics(history):
    # list all data in history
    print(history.keys())
    
    # summarize history for accuracy
    plt.subplots(1, 2, figsize=(14,6))
    plt.subplot(121)
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('Model accuracy',fontsize=20)
    plt.ylabel('Accuracy',fontsize=20)
    plt.xlabel('Epoch',fontsize=20)
    plt.legend(['train', 'test'], loc='center right',fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=18)
    
    # summarize history for loss
    plt.subplot(122)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss',fontsize=20)
    plt.ylabel('Loss',fontsize=20)
    plt.xlabel('Epoch',fontsize=20)
    plt.legend(['train', 'test'], loc='center right',fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.tick_params(axis='both', which='minor', labelsize=18)
    plt.show()

def load_data_field(path_home, name_dataset, name_subset, size_vlm):
    path_dataset = os.path.join(path_home, 'dataset', name_dataset, name_subset)
    path_seis = os.path.join(path_dataset, 'seis.dat')
    path_pred = os.path.join(path_dataset, 'pred.dat')    
    seis_vlm = np.fromfile(path_seis,dtype=np.single)
    pred_vlm = np.fromfile(path_pred,dtype=np.single) 
    seis_vlm = np.reshape(seis_vlm, size_vlm)
    pred_vlm = np.reshape(pred_vlm, size_vlm)
    return seis_vlm, pred_vlm

def load_data_synth(path_home, name_dataset, name_subset, name_file, size_vlm):
    path_dataset = os.path.join(path_home, 'dataset', name_dataset, name_subset)
    path_seis = os.path.join(path_dataset, 'seis', name_file)
    path_fault = os.path.join(path_dataset, 'fault', name_file)
    path_pred = os.path.join(path_dataset, 'pred', name_file)
    seis_vlm = np.fromfile(path_seis,dtype=np.single)
    fault_vlm = np.fromfile(path_fault,dtype=np.single)
    pred_vlm = np.fromfile(path_pred,dtype=np.single)
    seis_vlm = np.reshape(seis_vlm, size_vlm)
    fault_vlm = np.reshape(fault_vlm, size_vlm)
    pred_vlm = np.reshape(pred_vlm, size_vlm)
    return seis_vlm, fault_vlm, pred_vlm

class vlm_slicer_synth:
    def __init__(self, seis_vlm, fault_vlm, pred_vlm, idx=(63,63,63)):
        self.seis_vlm = seis_vlm
        self.fault_vlm = fault_vlm        
        self.pred_vlm = pred_vlm
        self.show_slices(idx)
        
    def create_img_alpha(self, img_input):
        img_alpha = np.zeros([np.shape(img_input)[0],np.shape(img_input)[1],4])
        # Yellow: (1,1,0), Red: (1,0,0)
        img_alpha[:,:,0] = 1
        img_alpha[:,:,1] = 0
        img_alpha[:,:,2] = 0       
        img_alpha[..., -1] = img_input
        return img_alpha
    
    def show_slices(self, idx, cmap_bg=plt.cm.gray_r):
        plt.figure()        
        for i in range(3):
            if i == 0:
                seis_slice = self.seis_vlm[:,:,idx[0]]                
                fault_slice = self.fault_vlm[:,:,idx[0]]
                pred_slice = self.pred_vlm[:,:,idx[0]]
                title = 'z-slice'
            elif i == 1:
                seis_slice = self.seis_vlm[:,idx[1],:]
                fault_slice = self.fault_vlm[:,idx[1],:]
                pred_slice = self.pred_vlm[:,idx[1],:]
                title = 'x-slice'
            elif i == 2:
                seis_slice = self.seis_vlm[idx[2],:,:]
                fault_slice = self.fault_vlm[idx[2],:,:]
                pred_slice = self.pred_vlm[idx[2],:,:]
                title = 'y-slice'

            for j in range(3):
                title_sub = ''
                index_ax = int(331+3*i+j)
                plt.subplot(index_ax)
                plt.imshow(seis_slice.T, cmap_bg)
                if j == 1:
                    title_sub = ' with label'
                    img_alpha = self.create_img_alpha(fault_slice.T)
                    plt.imshow(img_alpha, alpha=1, interpolation="bilinear")
                elif j == 2:
                    title_sub = ' with prediction'                    
                    img_alpha = self.create_img_alpha(pred_slice.T)
                    plt.imshow(img_alpha, alpha=1, interpolation="bilinear")
                plt.title(title + title_sub)
            plt.tight_layout()
        plt.show()
    
class vlm_slicer_interactive:
    def __init__(self, seis_vlm, pred_vlm, flag_slice):
        axis_color = 'lightgoldenrodyellow'
        idx_max = np.shape(seis_vlm)[2]
        self.seis_vlm = seis_vlm
        self.pred_vlm = pred_vlm
        self.cmap_bg=plt.cm.gray_r
        self.flag_slice = flag_slice
        
        fig = plt.figure()
        fig.subplots_adjust(left=0.25, bottom=0.25)
        self.ax = fig.add_subplot(111)
        self.idx_slider_ax = fig.add_axes([0.4, 0.1, 0.35, 0.03],
                                          facecolor=axis_color)        
        self.idx_slider = Slider(self.idx_slider_ax,'Z', 0, idx_max,
                                 valinit=0, valfmt='%d')
        self.idx_slider.on_changed(self.sliders_on_changed)
        self.reset_button_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])        
        self.reset_button = Button(self.reset_button_ax,
                                   'Reset', color=axis_color,
                                   hovercolor='0.975')
        self.reset_button.on_clicked(self.reset_button_on_clicked)
        self.plot_slice(idx=0)
        self.fig = fig
        self.imshow_alpha()
        plt.show()
    
    def plot_slice(self, idx):
        if   self.flag_slice == 0:
            self.seis_slice = self.seis_vlm[:,:,idx]
            self.pred_slice = self.pred_vlm[:,:,idx]
        elif self.flag_slice == 1:
            self.seis_slice = self.seis_vlm[:,idx,:]
            self.pred_slice = self.pred_vlm[:,idx,:]        
        elif self.flag_slice == 2:
            self.seis_slice = self.seis_vlm[idx,:,:]
            self.pred_slice = self.pred_vlm[idx,:,:]

    def create_img_alpha(self, img_input):
        img_alpha = np.zeros([np.shape(img_input)[0], np.shape(img_input)[1],4])
        threshold = 0.1
        img_input[img_input < threshold] = 0
        # Yellow: (1,1,0), Red: (1,0,0)
        img_alpha[:,:,0] = 1
        img_alpha[:,:,1] = 0
        img_alpha[:,:,2] = 0
        img_alpha[...,-1] = img_input
        return img_alpha
    
    def imshow_alpha(self,idx=0):
        self.plot_slice(idx)
        img_alpha = self.create_img_alpha(self.pred_slice.T)
        self.ax.imshow(self.seis_slice.T, self.cmap_bg)
        self.ax.imshow(img_alpha, alpha=0.5)   
    
    def sliders_on_changed(self, val):
        self.imshow_alpha(int(val))
        self.fig.canvas.draw_idle()

    def reset_button_on_clicked(self, mouse_event):
        self.idx_slider.reset()
