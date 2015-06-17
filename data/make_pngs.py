from os import listdir
from os.path import isfile, join
import paer

mypath = 'aedat/'
onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) and f.endswith('.aedat')]

for file in onlyfiles:
    ae = paer.aefile(mypath + str(file))
    aed= paer.aedata(ae).downsample((16,16))

    paer.create_pngs(aed, '16x16_' + str(file) + '_',path='more_images/temp',step=3000, dim=(16,16))


