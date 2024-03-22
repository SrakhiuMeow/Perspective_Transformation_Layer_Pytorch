import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import skimage


def gather_nd(params, indices):
    '''
    https://discuss.pytorch.org/t/how-to-do-the-tf-gather-nd-in-pytorch/6445/26
    4D example
    params: tensor shaped [n_1, n_2, n_3, n_4] --> 4 dimensional
    indices: tensor shaped [m_1, m_2, m_3, m_4, 4] --> multidimensional list of 4D indices
    
    returns: tensor shaped [m_1, m_2, m_3, m_4]
    
    ND_example
    params: tensor shaped [n_1, ..., n_p] --> d-dimensional tensor
    indices: tensor shaped [m_1, ..., m_i, d] --> multidimensional list of d-dimensional indices
    
    returns: tensor shaped [m_1, ..., m_1]
    '''

    out_shape = indices.shape[:-1]
    indices = indices.unsqueeze(0).transpose(0, -1) # roll last axis to fring
    ndim = indices.shape[0]
    indices = indices.long()
    idx = torch.zeros_like(indices[0], device=indices.device).long()
    m = 1
    
    for i in range(ndim)[::-1]:
        idx += indices[i] * m 
        m *= params.size(i)
    out = torch.take(params, idx)
    return out.view(out_shape)


class Perspective_Layer(nn.Module):
    def __init__(self, output_size=None, input_channel=3, tm=1, param=8, random_init=False, **kwargs):
        super(Perspective_Layer, self).__init__()
        self.output_size = output_size # (H, W)
        self.param = param
        self.tm = tm
        self.random_init = random_init        


        if self.random_init:
            initial_thetas = np.random.uniform(low=-1, high=1, size=(1, 8)).repeat(self.tm, axis=0)
        else:
            initial_thetas = np.array([[1, 0, 0, 0, 1., 0, 0., 0.]]).repeat(self.tm, axis=0)

        channel_tiles = [np.tile(initial_thetas[i],[input_channel,1]).astype('float32') for i in range(self.tm)]
        
        tx = np.stack(channel_tiles, axis=0)  # (tm, channel, param)
        
        self.wt_pers = nn.Parameter(torch.tensor(tx, dtype=torch.float32), requires_grad=True)  

    def compute_output_shape(self, input_shapes):
        if type(input_shapes) is list:
            input_shapes = input_shapes[0]
        else:
            input_shapes = input_shapes
        if self.output_size:
            H, W = self.output_size
        else:
            H, W = input_shapes[2], input_shapes[3]
        num_channels = input_shapes[1]
        return (None, num_channels, H, W)

        
    def forward(self, inputs):
        if isinstance(inputs, list):
            inp = inputs[0]
        else:
            inp = inputs

        expand_inp = inp.unsqueeze(0)
        tile_inputs = expand_inp.repeat(self.tm, 1, 1, 1, 1) # (tm, N, C, H, W)

        all_out = [self.vectorize_tms((self.wt_pers[i], tile_inputs[i])) for i in range(self.tm)] 
        all_out = torch.cat(all_out, dim=-1) # (N, C, Ho, Wo, tm)
        all_out = all_out.permute(0,1,4,2,3).flatten(1,2) # (N, C*tm, Ho, Wo)
        return all_out
    
    
    def vectorize_tms(self,args):
        wt_tms, inps = args # wt_tms: (C, param), inps: (N, C, H, W)
        batch_size = inps.shape[0]
        channels = inps.shape[1]
        channel_first_inp = inps.permute(1, 0, 2, 3) # (C, N, H, W)
        singleTM_out = [self.vectorize_out((wt_tms[i], channel_first_inp[i])) for i in range(channels)]
        singleTM_out = torch.stack(singleTM_out, dim=0) # (C, N, Ho, Wo, 1)
        # singleTM_out = singleTM_out.squeeze(-1)
        singleTM_out = singleTM_out.permute(1, 0, 2, 3, 4) # (N, C, Ho, Wo, 1)
        return singleTM_out

    def vectorize_out(self,arg):
        all_weights, inp = arg # all_weights: (param), inp: (N, H, W)
        # inp = inp.permute(2, 0, 1)
        inp2 = inp.unsqueeze(-1) # (N, H, W, 1)
        batch_size = inp2.shape[0] 
        w_expand = all_weights.unsqueeze(0) # (1, param)
        wt_tile = w_expand.repeat(batch_size, 1) # (N, param)
        out = self.apply_transformation(inp2, wt_tile, self.output_size) # (N, Ho, Wo, 1)
        return out

    def get_theta_matrix(self,theta):
        N = theta.shape[0]
        params_theta = theta.shape[1]
        identity_matrix = torch.eye(3)
        identity_params = identity_matrix.view(-1)
        remaining_params = identity_params[params_theta:]
        batch_tile_remaining = remaining_params.repeat(N)
        batch_params_remaining = [N, 9 - params_theta]
        batch_remaining = batch_tile_remaining.view(batch_params_remaining)
        theta_final = torch.cat([theta, batch_remaining], dim=1) # (N, 9)
        return theta_final
    
    
    def apply_transformation(self,features_inp, theta, out_shape=None, **kwargs):
        # features_inp (N, H, W, 1)
        # theta (N, 9)
        
        # get shapes of input features
        N = features_inp.shape[0]
        H = features_inp.shape[1]
        W = features_inp.shape[2]

        # get perspective transformation parameters
        pers_matrix = self.get_theta_matrix(theta)
        pers_matrix_shape = [N, 3, 3]
        pers_matrix = pers_matrix.view(pers_matrix_shape)

        # Grid generation
        if out_shape:
            Ho = out_shape[0]
            Wo = out_shape[1]
            x_s, y_s = self.generate_grids(Ho, Wo, pers_matrix) # (N, Ho, Wo)
        else:
            x_s, y_s = self.generate_grids(H, W, pers_matrix) # (N, H, W)

        features_out = self.interpolate(features_inp, x_s, y_s) # (N, Ho, Wo, 1)
        return features_out



    def extract_pixels(self,feature, x_vect, y_vect):
        # feature (N, H, W, 1)
        # x,y (N, H, W)
        N = x_vect.shape[0]
        H = x_vect.shape[1]
        W = x_vect.shape[2]

        batch_idx = torch.arange(0, N).view(N, 1, 1)
        b_tile = batch_idx.repeat(1, H, W) # (N, H, W)

        ind = torch.stack([b_tile, y_vect, x_vect], dim=3) # (N, H, W, 3)

        pixels_value_out = gather_nd(feature, ind) # (N, H, W)
        return pixels_value_out


    def generate_grids(self,H, W, theta):
        #get the batch size
        N = theta.size(0)

        # Meshgrid
        x = torch.linspace(-1.0, 1.0, W)
        y = torch.linspace(-1.0, 1.0, H)
        x_g, y_g = torch.meshgrid(x, y, indexing='ij')

        # Flatten the meshgrid
        flatten_x_g = x_g.flatten()
        flatten_y_g = y_g.flatten()

        # reshape to [x_g, y_g , 1] - (homogeneous form)
        ones = torch.ones_like(flatten_x_g)
        get_grid = torch.stack([flatten_x_g, flatten_y_g, ones])

        # get grid for each features in N
        get_grid = get_grid.unsqueeze(0)
        get_grid = get_grid.repeat(N, 1, 1)

        theta = theta.float()
        get_grid = get_grid.float()

        # use matmul to transform the sampling grid
        batch_grids = torch.matmul(theta, get_grid)

        #reshape
        batch_grids = batch_grids.view(N, 3, H, W)

        #  homogenous coordinates to Cartesian
        omega = batch_grids[:, 2, :, :]
        x_s = batch_grids[:, 0, :, :] / omega # (N, H, W)
        y_s = batch_grids[:, 1, :, :] / omega # (N, H, W)

        return x_s, y_s

    def interpolate(self,img, x, y):
        # img (N, H, W, 1)
        # x, y (N, Ho, Wo)
        H = img.shape[1]
        W = img.shape[2]
        max_y = float(H - 1)
        max_x = float(W - 1)
        zero = torch.zeros([], dtype=torch.int32)

        # rescale x and y to [0, W-1/H-1]
        x = x.float()
        y = y.float()
        x = 0.5 * ((x + 1.0) * (max_x - 1))
        y = 0.5 * ((y + 1.0) * (max_y - 1))

        # do sampling
        x0 = x.floor().int()
        x1 = x0 + 1
        y0 = y.floor().int()
        y1 = y0 + 1

        # clip
        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)

        # extract pixel value
        Ia = self.extract_pixels(img, x0, y0).unsqueeze(3)
        Ib = self.extract_pixels(img, x0, y1).unsqueeze(3)
        Ic = self.extract_pixels(img, x1, y0).unsqueeze(3)
        Id = self.extract_pixels(img, x1, y1).unsqueeze(3)


        # Finally calculate interpolated values
        # recast as float
        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()

        a = (x1_f - x) * (y1_f - y)
        b = (x1_f - x) * (y1_f - y0)
        c = (x - x0_f) * (y1_f - y)
        d = (x - x0_f) * (y - y0_f)
        wa = a.unsqueeze(3)
        wb = b.unsqueeze(3)
        wc = c.unsqueeze(3)
        wd = d.unsqueeze(3)

        # compute output
        output = wa * Ia + wb * Ib + wc * Ic + wd * Id # (N, Ho, Wo, 1)

        return output
    
if __name__ == '__main__':
    # Test
    input = torch.randn(8, 3, 448, 224)
    pers_layer = Perspective_Layer(output_size=(448, 224), tm=4, random_init=False)
    output = pers_layer(input)
    print(output.shape)
    # print(pers_layer.state_dict())
    # print(pers_layer.wt_pers.shape)
    # Output: torch.Size([1, 224, 224, 3])