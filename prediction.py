import os
import numpy as np
from keras.models import model_from_json
from obspy.io.segy.segy import _read_segy
from time import time
from tqdm import tqdm

def load_trained_model(path_home, name_model, name_weights):
    path_model = os.path.join(path_home, 'model')
    path_model_arch = os.path.join(path_model, name_model+ '.json')
    path_weights = os.path.join(path_model, 'weights', name_weights + '.h5')
    
    # load json and create model 
    json_file = open(path_model_arch, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(path_weights)
    print("Loaded model from disk")
    return model

def pred_subvlms(path_home, name_dataset, name_subset, model):
    size_vlm=np.array([1,1,1])*128
    path_dataset = os.path.join(path_home, 'dataset', name_dataset, name_subset)
    path_seis = os.path.join(path_dataset, 'seis')
    path_pred = os.path.join(path_dataset, 'pred')
    if not os.path.exists(path_pred):
        os.makedirs(path_pred)

    for file in tqdm(os.listdir(path_seis)):
        if file.endswith('.dat'):
            path_read = os.path.join(path_seis, file)
            path_write = os.path.join(path_pred, file)     
            seis_vlm = np.fromfile(path_read,dtype=np.single)
            seis_vlm_reshaped = np.reshape(seis_vlm,(1,*size_vlm,1))
            pred_vlm = model.predict(seis_vlm_reshaped)            
            fs = open(path_write, 'bw')
            pred_vlm.flatten().astype('float32').tofile(fs, format='%.4f')
            fs.close()

class pred_segy:
    def __init__(self, model, path_home, path_seis, name_dataset, name_subset, size_vlm, idx0_vlm):
        self.model = model
        self.size_subvlm=np.array([1,1,1])*128
        self.dataload_segy(path_seis, size_vlm, idx0_vlm)
        self.apply_trained_net()
        self.save_pred(path_home, name_dataset, name_subset)
        
    def dataload_segy(self, path_seis,size_vlm,idx0_vlm):
        t0 = time()
        print('Start Loading Segy Data')
        file_segy = _read_segy(path_seis).traces
        traces = np.stack(t.data for t in file_segy)
        inlines = np.stack(t.header.for_3d_poststack_data_this_field_is_for_in_line_number for t in file_segy)
        xlines = np.stack(t.header.for_3d_poststack_data_this_field_is_for_cross_line_number for t in file_segy)
        idx_inline = inlines - np.min(inlines)
        idx_xline = xlines - np.min(xlines)
        num_traces = len(traces)
        num_inline = len(np.unique(inlines))
        num_xline = len(np.unique(xlines))
        num_sample = len(file_segy[0].data)
        seis_vlm = np.zeros([num_inline, num_xline, num_sample])
        for i in range(num_traces):
            seis_vlm[idx_inline[i],idx_xline[i],:] = traces[i]
        t1 = time()
        print('\nCompleted Loading Segy Data')
        print('\nElapsed time: ' + "{:.2f}".format(t1-t0) + ' sec')
        seis_vlm = seis_vlm[idx0_vlm[0]:idx0_vlm[0]+size_vlm[0],
                            idx0_vlm[1]:idx0_vlm[1]+size_vlm[1],
                            idx0_vlm[2]:idx0_vlm[2]+size_vlm[2]]
        self.seis_vlm = seis_vlm

    def apply_trained_net(self):
        """ 
        a 3d array of gx[m1][m2][m3], please make sure the dimensions are correct!!!
        we strongly suggest to gain the seismic image before input it to the faultSeg!!!
        """
        size_vlm = np.shape(self.seis_vlm)
        n1, n2, n3 = self.size_subvlm[0],self.size_subvlm[1],self.size_subvlm[2]
        m1, m2, m3 = size_vlm[0],size_vlm[1],size_vlm[2]
    
        stdizer = lambda x: (x - np.mean(x)) / np.std(x)    
        os = 12 #overlap width
        c1 = np.round((m1+os)/(n1-os)+0.5)
        c2 = np.round((m2+os)/(n2-os)+0.5)
        c3 = np.round((m3+os)/(n3-os)+0.5)
        c1 = int(c1)
        c2 = int(c2)
        c3 = int(c3)
        p1 = (n1-os)*c1+os
        p2 = (n2-os)*c2+os
        p3 = (n3-os)*c3+os
        gx = np.reshape(self.seis_vlm,(m1,m2,m3))
        gp = np.zeros((p1,p2,p3),dtype=np.single)
        gy = np.zeros((p1,p2,p3),dtype=np.single)
        mk = np.zeros((p1,p2,p3),dtype=np.single)
        gs = np.zeros((1,*self.size_subvlm,1),dtype=np.single)
        gp[0:m1,0:m2,0:m3] = gx
        sc = self.getMask(os)
        for k1 in range(c1):
            for k2 in range(c2):
                for k3 in range(c3):
                    b1 = k1*n1-k1*os
                    e1 = b1+n1
                    b2 = k2*n2-k2*os
                    e2 = b2+n2
                    b3 = k3*n3-k3*os
                    e3 = b3+n3
                    gs[0,:,:,:,0]=gp[b1:e1,b2:e2,b3:e3]
                    gs = stdizer(gs)                
                    Y = self.model.predict(gs)
                    Y = np.array(Y)
                    gy[b1:e1,b2:e2,b3:e3] = gy[b1:e1,b2:e2,b3:e3]+Y[0,:,:,:,0]*sc
                    mk[b1:e1,b2:e2,b3:e3] = mk[b1:e1,b2:e2,b3:e3]+sc
        gy = gy/mk
        gy = gy[0:m1,0:m2,0:m3]
        self.pred_vlm = gy

    # set gaussian weights in the overlap bounaries
    def getMask(self, os):
        n1, n2, n3 = self.size_subvlm[0],self.size_subvlm[1],self.size_subvlm[2]
        sc = np.zeros(self.size_subvlm,dtype=np.single)
        sc = sc+1
        sp = np.zeros((os),dtype=np.single)
        sig = os/4
        sig = 0.5/(sig*sig)
        for ks in range(os):
            ds = ks-os+1
            sp[ks] = np.exp(-ds*ds*sig)
        
        for k1 in range(os):
            for k2 in range(n2):
                for k3 in range(n3):
                    sc[k1][k2][k3]=sp[k1]
                    sc[n1-k1-1][k2][k3]=sp[k1]
        for k1 in range(n1):
            for k2 in range(os):
                for k3 in range(n3):
                    sc[k1][k2][k3]=sp[k2]
                    sc[k1][n2-k2-1][k3]=sp[k2]
        for k1 in range(n1):
            for k2 in range(n2):
                for k3 in range(os):
                    sc[k1][k2][k3]=sp[k3]
                    sc[k1][k2][n3-k3-1]=sp[k3]
        return sc
 
    def save_pred(self, path_home, name_dataset, name_subset):
        path_dataset = os.path.join(path_home, 'dataset', name_dataset, name_subset)
        if not os.path.exists(path_dataset):
            os.makedirs(path_dataset)

        path_seis_w = os.path.join(path_dataset, 'seis.dat')
        path_pred_w = os.path.join(path_dataset, 'pred.dat')

        np.save(os.path.join(path_dataset,'seis.npy'),self.seis_vlm.astype('float32'))
        np.save(os.path.join(path_dataset,'pred.npy'),self.pred_vlm)
        
        f1 = open(path_seis_w, 'bw')
        f2 = open(path_pred_w, 'bw')        
        self.seis_vlm.flatten().astype('float32').tofile(f1, format='%.4f')
        self.pred_vlm.flatten().astype('float32').tofile(f2, format='%.4f')
        f1.close()
        f2.close()
