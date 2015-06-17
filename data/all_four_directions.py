import paer
import numpy as np

base_dir = 'aedat/'
file1 = 'left_to_right_1.aedat'
file2 = 'right_to_left_1.aedat'
file3 = 'top_to_bottom_1.aedat'
file4 = 'bottom_to_top_1.aedat'

# Each ball movement should be .5s long
animation_time = 0.2

# 3,280 events per second for 16*16 is reasonable for ball movement (might be even too high!)
num_events_p_s = 3280

# Helper function to read a file. Given (min,max) which are data ranges for extraction, this will return a cropped and
#  suitably sparse output.
def get_data(file, min, max, animation_time=animation_time, num_events=num_events_p_s*animation_time, offset=0):
    aefile = paer.aefile(file, max_events=max+1)
    aedata = paer.aedata(aefile)
    print 'Points: %i, Time: %0.2f. Sparsity: %i' % (len(aefile.data), (aefile.timestamp[-1]-aefile.timestamp[0])/1000000,
                                                  np.floor(len(aefile.data)/num_events))

    sparse = aedata[min:max].make_sparse(np.floor(len(aefile.data)/num_events))

    actual_time = (sparse.ts[-1]-sparse.ts[0])/1000000
    scale = actual_time/animation_time
    sparse.ts = (offset * 1000000) + np.round((sparse.ts-sparse.ts[0])/scale)
    # print sparse_ts[0], sparse_ts[-1], sparse_ts[-1]-sparse_ts[0], (sparse_ts[-1]-sparse_ts[0])/1000000

    return sparse

# Loop through all files - indexes are extrapolated.
d1 = get_data(base_dir+file1, 300000, 750000, offset=0*animation_time)
d2 = get_data(base_dir+file2, 300000, 600000, offset=1*animation_time)
d3 = get_data(base_dir+file3,  85000, 140000, offset=2*animation_time)
d4 = get_data(base_dir+file4,  65200, 131800, offset=3*animation_time)

# Need to pre-load a file, to get the correct headers when writing!
lib = paer.aefile(base_dir+file1, max_events=1)

final = paer.concatenate( (d1,d2,d3,d4) )
final_16 = final.downsample((16,16))

lib.save(final, 'aedat/200ms_all_dirs.aedat')
lib.save(final_16, 'aedat/200ms_16_16_all_dirs.aedat')

d1.downsample().save_to_mat('mat/200ms_16_16_norm_l2r.mat')
d2.downsample().save_to_mat('mat/200ms_16_16_norm_r2l.mat')
d3.downsample().save_to_mat('mat/200ms_16_16_norm_t2b.mat')
d4.downsample().save_to_mat('mat/200ms_16_16_norm_b2t.mat')