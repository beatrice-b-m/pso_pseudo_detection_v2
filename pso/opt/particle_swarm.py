# particle swarm optimization core
import numpy as np
from utility import viz
# from data.data_loader import Mammogram
import random
from sklearn.cluster import KMeans
from numba import njit


class Particle:
    """
    Particle class represents a solution inside a pool(Swarm).
    """

    def __init__(self, point, dim_shape, v_range):
        """
        Particle class constructor
        :param dim_shape: tuple(no_dim, )
            Shape of x(position), v(velocity).
        :param dim_ranges: tuple(double)
            Min and Max value(range) of dimension.
        :param v_range: tuple(double)
            Min and Max value(range) of velocity.
        :param dim_ranges: list(tuple(double)) e.g. [(200, 2400), (300, 3200), (100, 256)]
            Min and Max value(range) of velocity.
        """
        random.seed(4)
        np.random.seed(4)
        self.point = np.array(point)
        self.v = np.random.uniform(v_range[0], v_range[1], dim_shape)
        self.pbest = np.inf
        self.pbestpos = np.zeros(dim_shape, dtype=int)
        
        
        
class Cluster:
    def __init__(self, p_per_clust, center_coords, image_shape, dim_shape, 
                 min_bounds, max_bounds, v_range, iw_range, 
                 patch_width: int = 256):
        
        # patch params
        self.p_per_clust = p_per_clust
        self.patch_width = patch_width
        self.image_shape = image_shape
        
        # swarm params
        self.dim_shape = dim_shape
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds
        self.v_range = v_range
        self.iw_range = iw_range
        self.c1 = 0.0
        self.c2 = 0.0
        
        # set the max initial random shift to half the bound difference
        self.max_shift = (max_bounds - min_bounds) // 3
        
        # get initial point list
        point_list = self.get_init_point_list(center_coords)
                
        # assemble list of Particles and convert into an array
        self.p = np.array([Particle(point, dim_shape, self.v_range) for point in point_list])
        
        # vectorize functions
        self.get_pbest_arr = np.vectorize(self._get_pbest_arr)
        self.update_particle_velocities = np.vectorize(self._update_particle_velocities)
    
                
        # get initial cluster best
        self.cbestpos, self.cbest = self.update_cluster_bests(self.p)
        
    def get_init_point_list(self, center_coords, num_points: int = 8):
        center_coords = np.array(center_coords)
        point_list = []
        half_width = int(0.5*self.patch_width)
        
        # offset centroid coords to keep patches within image bounds
        centroid_coords = np.clip(center_coords.astype(int), 
                                  a_min = self.min_bounds + self.max_shift, 
                                  a_max = self.max_bounds - self.max_shift)
        
        center_point = np.clip(center_coords.astype(int), 
                               a_min = self.min_bounds, 
                               a_max = self.max_bounds)
        
        # append the first patch coordinates to the patch_coord_list
        point_list.append(center_point)
        
        # get number of random points around the centroid locations
        rand_points = self.p_per_clust - 1
        
        # get array of center points to shift
        point_center_array = np.full((rand_points, 2), centroid_coords)
        
        # get array of random values between -1 and 1
        rand_array = np.random.uniform(-1, 1, (rand_points, 2))
        
        # get array of maximum shifts
        shift_array = np.full((rand_points, 2), self.max_shift)
        
        # multiply rand_array and shift_array to get array of random shifts
        rand_shift_array = (rand_array * shift_array).astype(int)
        
        # apply random shifts to the points and concat them to the point_list
        point_list += (point_center_array + rand_shift_array).tolist()
        
        return point_list
    
    def update_cluster_bests(self, particles):
        # get particle bests array
        pbest_arr = self.get_pbest_arr(particles)
        pbest_arr = np.column_stack(pbest_arr)
        
        # get index of best cluster
        best_p = np.argmin(pbest_arr[:, -1])
        
        # get best fitness and position at index
        cbestpos = pbest_arr[best_p, :-1]
        cbest = pbest_arr[best_p, -1]
        
        return cbestpos, cbest
    
    def _get_pbest_arr(self, particle):
        # return an array of the positions and fitnesses of the cluster
        pos0_arr = particle.pbestpos[0]
        pos1_arr = particle.pbestpos[1]
        f_arr = particle.pbest
        return pos0_arr, pos1_arr, f_arr
    
    def _update_particle_velocities(self, p):
        # after evaluation, if topology is 'local'
        """
        It updates velocity of a particle.
        It is used by optimize function.
        :param p: Particle
        Particle to update velocity.
        :return: Particle
             Particle with updated velocity.
        """
        # update inertia weight
        iw = np.random.uniform(self.iw_range[0], self.iw_range[1], self.dim_shape)
        
        # update particle velocity
        p.v = iw * p.v + (self.c1 * np.random.uniform(0.0, 1.0, self.dim_shape) * (p.pbestpos - p.point)) + (
                          self.c2 * np.random.uniform(0.0, 1.0, self.dim_shape) * (self.cbestpos - p.point))
        
        # clip velocity within bounds
        p.v = np.clip(p.v, a_min=self.v_range[0], a_max=self.v_range[1])
        
        # update particle coords with velocity
        p.point = p.point + p.v

        # clip coords to be within bounds
        p.point = np.clip(p.point, a_min = self.min_bounds, a_max = self.max_bounds)
        return p


class Swarm:  # CAT = opt
    """
    Swarm class represents a pool of solution(particle).
    """

    def __init__(self, mammo, n_clusters, p_per_clust, dim_shape = (2,), patch_width = 256, 
                 max_v = 50, iw_range = (0.4, 0.9), c1_range = (0.5, 2.5), c2_range = (0.5, 2.5)):
        
        random.seed(4)
        np.random.seed(4)
        
        self.mammo = mammo
        self.dim_shape = dim_shape
        self.v_range = (-max_v, max_v)
        self.c1_range = c1_range
        self.c2_range = c2_range
        self.iw_range = iw_range
        self.c1 = 0.0
        self.c2 = 0.0
        
        # get image bounds
        self.half_width = int(0.5*patch_width)
        self.min_bounds = np.full(self.dim_shape, 
                                  self.half_width, 
                                  dtype=int)
        self.max_bounds = np.full(self.dim_shape, 
                                  [mammo.crop_shape[0]-self.half_width, 
                                   mammo.crop_shape[1]-self.half_width], 
                                  dtype=int)
        
        # get the center points for the n_clusters
        cluster_center_list = self.get_init_coords(n_clusters = n_clusters)
        
        # initialize clusters
        self.clusters = [Cluster(p_per_clust, 
                                 coords, 
                                 mammo.crop_shape,  
                                 dim_shape, 
                                 self.min_bounds, 
                                 self.max_bounds, 
                                 self.v_range, 
                                 self.iw_range) for coords in cluster_center_list]
        
        self.swarm_pred_log = np.empty((0, 3), dtype=np.float32)
        
        # vectorize functions
        self.get_cbest_arr = np.vectorize(self._get_cbest_arr)
        self.get_particle_array = np.vectorize(self._get_particle_array)
        self.update_particle_bests = np.vectorize(self._update_particle_bests)
        self.get_particle_locs = np.vectorize(self._get_particle_locs)
        self.update_particle_velocities = np.vectorize(self._update_particle_velocities)
        
        # get initial particle array
        self.p_array = self.prepare_particle_array()
        
        # get initial global bests
        self.gbestpos, self.gbest = self.update_global_bests()
        
    def get_init_coords(self, mode: str = 'intensity', n_clusters: int = 8):
        """
        :params mode: str
        'intensity' for intensity clustering method
        'random' for random point dist
        """
        img = self.mammo.crop_img[:, :, 0]
        
        if mode == 'intensity':
            # calculate the nth percentile value
            percent = np.percentile(img, 95)

            # create a mask where pixel values are higher than the percentile cutoff
            thresh_img = (img > percent)

            # get the coordinates associated with each point
            coords = np.argwhere(thresh_img)

            # use KMeans clustering algorithm to find 8 cluster centers
            clustering = KMeans(n_clusters = n_clusters, max_iter = 200, n_init=10, random_state = 4).fit(coords)
            clust_centers = clustering.cluster_centers_
            
        elif mode == 'random':
            # generate random points within bounds
            clust_centers = np.random.randint(self.min_bounds, 
                                              self.max_bounds, 
                                              (n_clusters, 2), 
                                              dtype=int)
        return clust_centers.tolist()
            
        
    def assemble_patches(self, p_array):
        # get array of particle locations from p_array
        p_locs = np.column_stack(self.get_particle_locs(self.p_array))
        
        # get list of patch coordinates
        patch_coord_list = self.get_patch_coords(p_locs)
        
        w = int(2*self.half_width)
        patches = self.get_patches(patch_coord_list, w, self.mammo.crop_img)
        pad_patches = self.pad_patches(patches)
        
        return pad_patches, patches, p_locs
    
    def get_patch_coords(self, patch_centers):
        # Create a new array with the same shape as input_array
        width_array = np.full_like(patch_centers, self.half_width)

        # offset coords by half the patch width and concat the results
        patch_coords = np.hstack([patch_centers - width_array, 
                                  patch_centers + width_array])
        return patch_coords.astype(int)
    
    @staticmethod
    @njit
    def get_patches(patch_coord_arr, width, img):
        # initialize patch array
        patches = np.empty((patch_coord_arr.shape[0], 
                            width, 
                            width, 3), 
                           dtype = np.uint8)
        
        # build patch array
        for i, coords in enumerate(patch_coord_arr):
            patches[i, :, :, :] = img[coords[0]:coords[2], 
                                      coords[1]:coords[3], :]
        return patches
    
    def pad_patches(self, patches, out_width = 512):
        pad_array = np.zeros((patches.shape[0], out_width, out_width, 3), dtype=np.uint8)
        offset = int((out_width - patches.shape[1]) / 2)
        
        pad_array[:, offset:(out_width-offset), offset:(out_width-offset), :] = patches
        return pad_array
        
    def _update_particle_bests(self, p, fitness):
        # apply to p_array, updates bests after evaluation
        """
        It updates particle position.
        :param p: Particle
            Particle to updated position.
        :param fitness: double
            Fitness value or loss(to be optimized).
        :return: Particle
            Updated Particle.
        """
        if fitness < p.pbest:
            p.pbest = fitness
            p.pbestpos = p.point
        return p
    
    def update_global_bests(self):
        # get particle bests array
        cbest_arr = self.get_cbest_arr(self.clusters)
        cbest_arr = np.column_stack(cbest_arr)
        
        # get index of best cluster
        best_c = np.argmin(cbest_arr[:, -1])
        
        # get best fitness and position at index
        gbestpos = cbest_arr[best_c, :-1]
        gbest = cbest_arr[best_c, -1]
        
        return gbestpos, gbest
    
    def _get_cbest_arr(self, cluster):
        # return an array of the positions and fitnesses of the cluster
        pos0_arr = cluster.cbestpos[0]
        pos1_arr = cluster.cbestpos[1]
        f_arr = cluster.cbest
        return pos0_arr, pos1_arr, f_arr
    
    def _get_particle_array(self, cluster):
        # return a nested array of all particles from each cluster
        out_arr = cluster.p
        return out_arr
    
    def _get_particle_locs(self, particle):
        # return an array of particle center coords
        out0, out1 = particle.point
        return out0, out1
    
    def prepare_particle_array(self):        
        # get nested array
        particle_array = self.get_particle_array(self.clusters)
        
        # unnest array
        particle_array = np.vstack(particle_array)
        pa_shape = particle_array.shape
        
        # reshape array
        particle_array = particle_array.reshape(pa_shape[0]*pa_shape[1])
        return particle_array
    
    # AFTER EVALUATION
    def update_swarm(self, particle_array, fitness, topology: str = 'local'):
        # update particle bests based on fitness
        particle_array = self.update_particle_bests(particle_array, fitness)
        
        # update cluster bests
        for cluster in self.clusters:
            cluster.cbestpos, cluster.cbest = cluster.update_cluster_bests(cluster.p)
        
        # update swarm bests
        self.gbestpos, self.gbest = self.update_global_bests()
        
        # update particle velocities/positions
        if topology == 'global':
            # update particle velocities with global best
            self.p_array = self.update_particle_velocities(self.p_array)
            
        elif topology == 'local': # if topology is 'local'
            [cluster.update_particle_velocities(cluster.p) for cluster in self.clusters]
            
        else:
            print(f"Unrecognized topology '{topology}'")
            
            
        
    def _update_particle_velocities(self, p):
        # after evaluation
        """
        It updates velocity of a particle.
        It is used by optimize function.
        :param p: Particle
        Particle to update velocity.
        :return: Particle
             Particle with updated velocity.
        """
        # update inertia weight
        iw = np.random.uniform(self.iw_range[0], self.iw_range[1], self.dim_shape)
        
        # update particle velocity
        p.v = iw * p.v + (self.c1 * np.random.uniform(0.0, 1.0, self.dim_shape) * (p.pbestpos - p.point)) + (
                          self.c2 * np.random.uniform(0.0, 1.0, self.dim_shape) * (self.gbestpos - p.point))
        
        # clip velocity within bounds
        p.v = np.clip(p.v, a_min=self.v_range[0], a_max=self.v_range[1])
        
        # update particle coords with velocity
        p.point = p.point + p.v

        # clip coords to be within bounds
        p.point = np.clip(p.point, a_min = self.min_bounds, a_max = self.max_bounds)
        return p

    def update_c(self, total_it, curr_it):
        c1_range = self.c1_range
        c2_range = self.c2_range
        
        # update cognitive and social params
        self.c1 = ((c1_range[0] - c1_range[1]) * (curr_it / total_it)) + c1_range[1]
        self.c2 = ((c2_range[1] - c2_range[0]) * (curr_it / total_it)) + c2_range[0]
        
        # update cluster params
        for cluster in self.clusters:
            cluster.c1 = self.c1
            cluster.c2 = self.c2
            
    def correct_background(self, patches, fitness, pv_thresh = 5, bg_thresh = 0.7):
        # get mask of pixels lower than the pixel value threshold
        bg_mask = patches < pv_thresh

        # get the proportion of background pixels along axis 0
        bg_percent = np.mean(bg_mask, axis=(1, 2, 3))

        # where the bg_percent is higher than the threshold, replace the fitness with 0
        fitness = np.where(bg_percent > bg_thresh, 0, fitness)
        return fitness

    def optimize(self, function, topology, n_it: int, 
                 method: str, i: int, plot_its: bool = False, save: bool = False):

        it_viz = False

        # if maximizing, optimize the negative fitness instead
        inverse = 1
        if method == 'max':
            inverse = -1
        
        # loop through particle swarm optimization iterations
        for j in range(n_it):
            # update social and cognitive parameters for current iteration
            self.update_c(n_it, j)

            # get patches
            pad_patches, patches, patch_centers = self.assemble_patches(self.p_array)
            
            # get patch predictions for swarm
            fitness = function(pad_patches)
            fitness = fitness * inverse
            
            # apply background correction to predictions
            fitness = self.correct_background(patches, fitness)
            
            # create log array
            log_array = np.hstack((patch_centers, fitness.reshape(-1, 1)))
            
            # append current swarm predictions to self.swarm_pred_log
            self.swarm_pred_log = np.vstack((self.swarm_pred_log, log_array))
            
            if plot_its:
                # visualize here
                viz.plot_swarm(log_array, self.mammo, self.half_width, j=j, save=save)
            
            # update swarm bests, velocities, and positions
            self.update_swarm(self.p_array, fitness, topology = topology)

            print("Iteration: ", j, " gbest: ", self.gbest)

        return self.swarm_pred_log
