import numpy as np
import keras
from itertools import combinations
from scipy.signal import butter, filtfilt
from scipy.interpolate import RegularGridInterpolator
import bruges
from tqdm import tqdm
import os

class DefineParams():
    """ Parameters for Creating Synthetic Traces """
    def __init__(self, num_data, patch_size):
        size_tr = 200
        nx, ny, nz = ([patch_size]*3)
        nxy = nx*ny
        nxyz = nxy*nz
        nx_tr, ny_tr, nz_tr = ([size_tr]*3)
        nxy_tr = nx_tr*ny_tr
        nxyz_tr = nxy_tr*nz_tr
        x = np.linspace(0, nx_tr-1, nx_tr)
        y = np.linspace(0, nx_tr-1, ny_tr)
        z = np.linspace(0, nz_tr-1, nz_tr)
        xy = np.reshape(np.array([np.meshgrid(x, y, indexing='ij')]), [2, nxy_tr]).T
        xyz = np.reshape(np.array([np.meshgrid(x, y, z, indexing='ij')]), [3, nxyz_tr]).T
        
        ' Feature Size '
        self.nx = nx                        # Height of input feature
        self.ny = ny                        # Width of input feature
        self.nz = nz                        # Number of classes
        self.nxy = nxy                      # 
        self.nxyz = nxyz                    # 
        self.num_data = num_data
        
        ' Synthetic traces '
        self.dt = 0.004                     # Synthetic Traces: Sampling interval (ms)
        self.x = x                          # 
        self.y = y                          # 
        self.z = z                          # 
        self.xy = xy                        # 
        self.xyz = xyz                      # 
        self.x0 = int(nx/2)                 # Synthetic Traces: x center
        self.y0 = int(ny/2)                 # Synthetic Traces: y center
        self.z0 = int(nz/2)                 # Synthetic Traces: z center
        self.nx_tr = nx_tr                  # Synthetic Traces: sampling in x
        self.ny_tr = ny_tr                  # Synthetic Traces: sampling in y
        self.nz_tr = nz_tr                  # Synthetic Traces: sampling in z
        self.nxy_tr = nxy_tr                # 
        self.nxyz_tr = nxyz_tr              # 
        self.x0_tr = int(nx_tr/2)           # Synthetic Traces: x center
        self.y0_tr = int(ny_tr/2)           # Synthetic Traces: y center
        self.z0_tr = int(nz_tr/2)           # Synthetic Traces: z center
        self.lcut = 5                       # Bandpass filter: Lower cutoff
        self.hcut = 80                      # Bandpass filter: Upper cutoff
        self.t_lng = 0.082                  # Ricker wavelet: Length
        
        ' Ranges for random parameters'
        self.a_rng = (0,1)                  # Sinusoidal deformation: Amplitude
        self.b_rng = (0,5)                  # Sinusoidal deformation: Frequency
        self.c_rng = (0,nx_tr-1)            # Sinusoidal deformation: Initial phase
        self.d_rng = (0,ny_tr-1)            # Linear deformation: Slope
        self.sigma_rng = (10,30)            # 

        self.e_rng = (0,1)                  # Linear deformation: Intercept
        self.f_rng = (0,0.1)                # Ricker wavelet: Central frequency
        self.g_rng = (0,0.1)                # Ricker wavelet: Central frequency
        
        self.x0_rng = (32,nx-1-32)
        self.y0_rng = (32,ny-1-32)
        self.z0_rng = (32,nz-1-32)
        self.num_flt_rng = (5,8)
        self.throw_rng = (5,30)             # Fault: Displacement
        self.dip_rng = (62,82)
        self.strike_rng = (0,360)

        self.snr_rng = (2,5)                # Signal Noise Ratio
        self.f0_rng = (20,35)               # 

class GenerateParams:
    def __init__(self, prm):
        # Deformation Parameters
        num_gauss= 4
        self.param_deform(prm, num_gauss)

        # Signal Noise Ratio
        self.snr = self.randomize_prm(prm.snr_rng)
        self.f0 = self.randomize_prm(prm.f0_rng)

    def randomize_prm(self, lmts, n_rand=1, pos_neg=False):
        if pos_neg:
            coeff = np.random.choice([-1, 1])
        else:
            coeff = 1
        prm = coeff * np.random.uniform(lmts[0], lmts[1], n_rand)
        return prm
    
    def param_deform(self, prm, num_gauss):
        # 2D Gaussian Deformation
        self.a = self.randomize_prm(prm.a_rng, pos_neg=True)
        self.b = self.randomize_prm(prm.b_rng, num_gauss, pos_neg=True)
        self.c = self.randomize_prm(prm.c_rng, num_gauss)
        self.d = self.randomize_prm(prm.d_rng, num_gauss)
        self.sigma = self.randomize_prm(prm.sigma_rng, num_gauss)

        # Planar Deformation
        self.e = self.randomize_prm(prm.e_rng, pos_neg=True)
        self.f = self.randomize_prm(prm.f_rng, pos_neg=True)
        self.g = self.randomize_prm(prm.g_rng, pos_neg=True)

    def param_fault(self, prm):
        # Fault Throw
        min_dist = 20 # Minimum distance between faults
        num_flt = np.random.randint(prm.num_flt_rng[0],prm.num_flt_rng[1]+1,1)[0]
        self.x0_f, self.y0_f, self.z0_f = self.pick_fault_center(prm, num_flt, min_dist)
        self.throw = self.randomize_prm(prm.throw_rng, num_flt, pos_neg=True)
        self.dip = self.randomize_prm(prm.dip_rng, num_flt)
        self.strike = self.randomize_prm(prm.strike_rng, num_flt)
        self.type_flt = np.random.randint(0, 1+1, num_flt) # 0: Linear, 1: Gaussian
        
    def pick_fault_center(self, prm, num_flt, min_dist):
        def dist_multi_pts(coords, n_rand, min_dist):
            dist_cal = lambda x1,y1,z1,x2,y2,z2: np.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)
            comb = combinations(np.arange(n_rand),2)
            dist = []
            for i in list(comb):
                coords1 = coords[i[0]]
                coords2 = coords[i[1]]
                dist.append(dist_cal(coords1[0],coords1[1],coords1[2],
                                     coords2[0],coords2[1],coords2[2]))
            dist = np.array(dist)
            flag_dist = sum(np.array(dist) > min_dist) > 0
            return flag_dist, dist
        
        flag_dist = False
        while flag_dist == False:
            x0_f = self.randomize_prm(prm.x0_rng, num_flt)
            y0_f = self.randomize_prm(prm.y0_rng, num_flt)
            z0_f = self.randomize_prm(prm.z0_rng, num_flt)
            if num_flt == 1:
                flag_dist = True
            else:
                coords = [tuple([x0_f[i], y0_f[i], z0_f[i]]) for i in range(num_flt)]
                flag_dist, _ = dist_multi_pts(coords, num_flt, min_dist)
        return x0_f, y0_f, z0_f
    
    def return_prm_list(self):
        prm_list = [self.a, self.b, self.c, self.d, self.sigma,
                    self.e, self.f, self.g, self.x0_f, self.y0_f, self.z0_f,
                    self.throw, self.dip, self.strike, self.snr, self.f0]
        return prm_list

class CreateSynthRefl(GenerateParams):
    def __init__(self, prm):
        super().__init__(prm)
        self.refl = np.zeros([prm.nx_tr, prm.ny_tr, prm.nz_tr])
        self.labels = np.zeros(prm.nxyz_tr)
        self.create_1d_model(prm)
        self.deformation(prm)
        
        flag_zero_counts = False
        while flag_zero_counts == False:
            self.param_fault(prm)
            self.throw_shift(prm)
            flag_zero_counts = self.zero_counts(prm)
    
    def zero_counts(self, prm):
        xyz = np.reshape(self.refl, [prm.nx_tr, prm.ny_tr, prm.nz_tr])
        xyz_crop = xyz[prm.x0_tr-prm.x0:prm.x0_tr+prm.x0,
                       prm.y0_tr-prm.y0:prm.y0_tr+prm.y0,
                       prm.z0_tr-prm.z0:prm.z0_tr+prm.z0].flatten()
        if sum(xyz_crop == 0) == 0:
            flag_zero_counts = True
        else:
            flag_zero_counts = True
        return flag_zero_counts

    def create_1d_model(self, prm):
        ''' Create 1D synthetic reflectivity model '''
        num_rand = int(prm.nz_tr*0.5)
        idx_refl = np.random.randint(0, prm.nz_tr, num_rand)
        refl = np.zeros(prm.nz_tr)
        refl[idx_refl] = 2*np.random.rand(num_rand)-1
        self.refl = np.tile(refl,[prm.nxy_tr,1])
        
    def deformation(self, prm):
        """
            Apply 2D Gaussian and Planar deformation.
            Computation is parallelized on GPU using cupy.
        """
        import cupy as cp
        xy_cp = cp.asarray(prm.xy)
        a_cp = cp.asarray(self.a)
        b_cp = cp.asarray(self.b)
        c_cp = cp.asarray(self.c)
        d_cp = cp.asarray(self.d)
        sigma_cp = cp.asarray(self.sigma)
        e_cp = cp.asarray(self.e)
        f_cp = cp.asarray(self.f)
        g_cp = cp.asarray(self.g)
        z_cp = cp.asarray(prm.z)        
        
        func_planar = cp.ElementwiseKernel(
                in_params='T x, T y, T e, T f, T g',
                out_params='T z',
                operation=\
                '''
                z = e + f*x + g*y;
                ''',
                name='func_planar'
                )

        func_gauss2d = cp.ElementwiseKernel(
                in_params='T x, T y, T b, T c, T d, T sigma',
                out_params='T z',
                operation=\
                '''
                z = b*expf(-(powf(x-c,2) + powf(y-d,2))/(2*powf(sigma,2)));
                ''',
                name='func_gauss2d'
                )

        gauss_2d_cp = cp.zeros_like(xy_cp[:,0])            
        for i in range(len(self.b)):
            gauss_2d_cp += func_gauss2d(xy_cp[:,0],xy_cp[:,1],b_cp[i],c_cp[i],d_cp[i],sigma_cp[i])
        s1_cp = a_cp +(1.5/z_cp)*cp.outer(cp.transpose(gauss_2d_cp),z_cp)
        s2_cp = func_planar(xy_cp[:,0],xy_cp[:,1],e_cp,f_cp,g_cp)

        refl_cp = cp.asarray(self.refl)     
        for i in range(prm.nxy_tr):
            s = s1_cp[i,:]+s2_cp[i]+z_cp
            mat = cp.tile(z_cp,(len(s),1)) - cp.tile(cp.expand_dims(s,1),(1,len(z_cp)))
            refl_cp[i,:] = cp.dot(refl_cp[i,:], cp.sinc(mat))

        self.refl = np.reshape(cp.asnumpy(refl_cp), [prm.nxy_tr, prm.nz_tr])
    
    def throw_shift(self, prm):
        """ Add fault throw with linear and gaussian offset """
        def z_proj(x, y, z, x0_f, y0_f, z0_f, theta, phi):
            x1 = x0_f+(prm.nx_tr-prm.nx)/2
            y1 = y0_f+(prm.ny_tr-prm.ny)/2
            z1 = z0_f+(prm.nz_tr-prm.nz)/2
            z_flt_plane = z1+(np.cos(phi)*(x-x1)+np.sin(phi)*(y-y1))*np.tan(theta)
            return z_flt_plane

        def fault_throw(theta, phi, throw, z0_f, type_flt, prm):
            """ Define z shifts"""
            z1 = (prm.nz_tr-prm.nz)/2+z0_f
            z2 = (prm.nz_tr-prm.nz)/2+prm.nz
            z3 = (prm.nz_tr-prm.nz)/2
            if type_flt == 0:     # Linear offset
                if throw > 0:     # Normal fault
                    z_shift = throw*np.cos(theta)*(prm.z-z1)/(z2-z1)
                    z_shift[z_shift < 0] = 0
                else:             # Reverse fault
                    z_shift = throw*np.cos(theta)*(prm.z-z1)/(z3-z1)
                    z_shift[z_shift > 0] = 0
            else:                 # Gaussian offset
                gaussian1d = lambda z, sigma: throw*np.sin(theta)*np.exp(-(z-z1)**2/(2*sigma**2))
                z_shift = gaussian1d(prm.z, sigma=20)

            """ flag offset """
            flag_offset = np.zeros([prm.nxy_tr, prm.nz_tr], dtype=bool)
            for i in range(prm.nxy_tr):
                flag_offset[i,:] = np.abs(z_shift) > 1
            flag_offset = np.reshape(flag_offset, prm.nxyz_tr)
            return z_shift, flag_offset

        def replace(xyz0, idx_repl, x1, y1, z1, prm):
            """ Replace """
            xyz1 = np.reshape(xyz0.copy(),[prm.nx_tr,prm.ny_tr,prm.nz_tr])
            func_3d_interp = RegularGridInterpolator((prm.x, prm.y, prm.z), xyz1, method='linear',
                                                     bounds_error=False, fill_value=0)
            idx_interp = np.reshape(idx_repl, prm.nxyz_tr)
            xyz1 = np.reshape(xyz1,prm.nxyz_tr)
            xyz1[idx_interp] = func_3d_interp((x1[idx_interp],y1[idx_interp],z1[idx_interp]))
            return xyz1

        flag_zero_counts = False
        while flag_zero_counts == False:
            self.param_fault(prm)            
            for i in range(len(self.throw)):
                # z values on a fault plane
                theta = self.dip[i] / 180 * np.pi
                phi = self.strike[i] / 180 * np.pi
                x, y, z = prm.xyz[:,0], prm.xyz[:,1], prm.xyz[:,2]
                z_flt_plane = z_proj(x, y, z, self.x0_f[i], self.y0_f[i], self.z0_f[i], theta, phi)
                idx_repl = prm.xyz[:,2] <= z_flt_plane
                z_shift, flag_offset = \
                    fault_throw(theta, phi, self.throw[i], self.z0_f[i], self.type_flt[i], prm)
                x1 = prm.xyz[:,0] - np.tile(z_shift, prm.nxy_tr)*np.cos(theta)*np.cos(phi)
                y1 = prm.xyz[:,1] - np.tile(z_shift, prm.nxy_tr)*np.cos(theta)*np.sin(phi)
                z1 = prm.xyz[:,2] - np.tile(z_shift, prm.nxy_tr)*np.sin(theta)
    
                # Fault throw
                refl = self.refl.copy()
                refl = replace(refl, idx_repl, x1, y1, z1, prm)
                self.refl = np.reshape(refl, [prm.nxy_tr, prm.nz_tr])
    
                # Fault Label
                labels = self.labels.copy()
                if i > 0:                
                    labels = replace(labels, idx_repl, x1, y1, z1, prm)
                    labels[labels > 0.4] = 1
                    labels[labels <= 0.4] = 0                
                flt_flag = (0.5*np.tan(self.dip[i]/180*np.pi)>abs(z-z_flt_plane)) & flag_offset
                labels[flt_flag] = 1
                self.labels = labels
            flag_zero_counts = self.zero_counts(prm)

class CreateSyntheticTrace(CreateSynthRefl):
    def __init__(self, prm):
        super().__init__(prm)
        self.traces = np.zeros([prm.nxy_tr, prm.nz_tr])
        self.convolve_wavelet(prm)
        self.add_noise(prm)
        self.crop_center_patch(prm)
        self.standardizer()
        self.traces = np.reshape(self.traces, [prm.nx, prm.ny, prm.nz])
        self.labels = np.reshape(self.labels, [prm.nx, prm.ny, prm.nz])
        
    def convolve_wavelet(self, prm):
        ''' Convolve reflectivity model with a Ricker wavelet '''
        wl = bruges.filters.wavelets.ricker(prm.t_lng, prm.dt, self.f0)
        for i in range(prm.nxy_tr):
            self.traces[i,:] = np.convolve(self.refl[i,:], wl, mode='same')
    
    def add_noise(self, prm):
        ''' Add some noise to traces to imitate real seismic data '''
        order = 5
        nyq = 1 / prm.dt / 2
        low = prm.lcut / nyq
        high = prm.hcut / nyq
        b, a = butter(order, [low, high], btype='band')
        for i in range(prm.nxy_tr):
            noise = bruges.noise.noise_db(self.traces[i,:], self.snr)
            self.traces[i,:] = filtfilt(b, a, self.traces[i,:] + noise)
        
    def crop_center_patch(self, prm):
        ''' Extract the central part in the input size '''
        def func_crop(xyz):
            xyz = np.reshape(xyz, [prm.nx_tr, prm.ny_tr, prm.nz_tr])
            xyz_crop = xyz[prm.x0_tr-prm.x0:prm.x0_tr+prm.x0,
                           prm.y0_tr-prm.y0:prm.y0_tr+prm.y0,
                           prm.z0_tr-prm.z0:prm.z0_tr+prm.z0]
            return np.reshape(xyz_crop, [prm.nxy, prm.nz])
        self.traces = func_crop(self.traces)
        self.labels = np.reshape(self.labels, [prm.nxy_tr, prm.nz_tr])        
        self.labels = func_crop(self.labels)
    
    def standardizer(self):
        ''' Standardize amplitudes within the image '''
        std_func = lambda x: (x - np.mean(x)) / np.std(x)
        tr_std = std_func(self.traces)
        tr_std[tr_std > 1] = 1
        tr_std[tr_std < -1] = -1
        self.traces = tr_std
    
def SyntheticSeisGen(path, num_data, patch_size=128):
    # Create randome synthetic images
    prm = DefineParams(num_data, patch_size)    
    path_seis = path + '/seis/'
    path_fault = path + '/fault/'
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(path_seis)
        os.makedirs(path_fault)

    for i in tqdm(range(num_data)):
        SynTr = CreateSyntheticTrace(prm)        
        fs = open(path_seis + str(i)+'.dat', 'bw')
        fl = open(path_fault + str(i)+'.dat', 'bw')
        SynTr.traces.flatten().astype('float32').tofile(fs, format='%.4f')
        SynTr.labels.flatten().astype('float32').tofile(fl, format='%.4f')

class DataGenerator(keras.utils.Sequence):
    'Generates data for keras'
    def __init__(self, dpath, fpath, data_IDs, batch_size=1, num_data_aug=4,
                 dim=(128,128,128), n_channels=1, shuffle=True):
        'Initialization'
        self.dim   = dim
        self.dpath = dpath
        self.fpath = fpath
        self.batch_size = batch_size
        self.num_data_aug = num_data_aug
        self.data_IDs   = data_IDs
        self.n_channels = n_channels
        self.shuffle    = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data_IDs)/self.batch_size))

    def __getitem__(self, index):
        'Generates one batch of data'
        # Generate indexes of the batch
        bsize = self.batch_size
        indexes = self.indexes[index*bsize:(index+1)*bsize]

        # Find list of IDs
        data_IDs_temp = [self.data_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(data_IDs_temp)
        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, data_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.zeros((self.num_data_aug, *self.dim, self.n_channels),dtype=np.single)
        Y = np.zeros((self.num_data_aug, *self.dim, self.n_channels),dtype=np.single)
        gx  = np.fromfile(os.path.join(self.dpath,str(data_IDs_temp[0])+'.dat'),dtype=np.single)
        fx  = np.fromfile(os.path.join(self.fpath,str(data_IDs_temp[0])+'.dat'),dtype=np.single)
        gx = np.reshape(gx,self.dim)
        fx = np.reshape(fx,self.dim)
        # data augmentation
        idx_rot = np.random.randint(0, 4, self.num_data_aug)
        for i in range(self.num_data_aug):
            X[i,] = np.reshape(np.rot90(gx,idx_rot[i],(0,1)), (*self.dim,self.n_channels))
            Y[i,] = np.reshape(np.rot90(fx,idx_rot[i],(0,1)), (*self.dim,self.n_channels))  
        return X,Y