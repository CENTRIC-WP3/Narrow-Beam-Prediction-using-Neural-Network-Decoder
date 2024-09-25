import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
# codebook = ['wideBeam_linear_8beams', 'wideBeam_linear_shift_8beams','wideBeam_hybrid_8beams','wideBeam_linear_8beams_comparenew','wideBeam_linear_shift_8beams_comparenew','wideBeam_hybrid_8beams_comparenew']
# label=['WB with ML', 'CS-WB with ML', 'PR-WB with ML','WB with DD', 'CS-WB with DD', 'PR-WB with DD']
# label=['Fixed 64 beam pairs', 'Fixed 32 beam pairs', 'Fixed 16 beam pairs', 'Random 32-32 beam pairs', 'Random [32,64] -32 beam pairs', 'Random 16-16 beam pairs', 'Random [16,64]-32 beam pairs', 'Random [16,64]-16 beam pairs']
label=['Random 32-32 Tx beam', 'Random [32,64] -32 Tx beam', 'Random 16-16 Tx beam', 'Random [16,64]-32 Tx beam', 'Random [16,64]-16 Tx beam']

#tx-rx
# configuration = [['3','4','4','uniform','4'],
#                  ['3','8','8','uniform','8'],
#                  ['3','16','16','uniform','16'],
#                  ['3','8','8','random','8'],
#                  ['3','8','8','random','random'],
#                  ['3','16','16','random','16'],
#                  ['3','16','8','random','random'],
#                  ['3','16','16','random','random']]
#tx
configuration = [['0','2','2','random','2'],
                 ['0','2','2','random','random'],
                 ['0','4','4','random','4'],
                 ['0','4','2','random','random'],
                 ['0','4','4','random','random']]

# line = ['g-','g^-','g*-.','b-','b^-','r-','r^-','r*-.']
line = ['g-','g^-','r-','r^-','r*-.']

# codebook = ['wideBeam_linear_shift_8beams_comparenew','wideBeam_hybrid_8beams_comparenew']
# label=[ 'CS-WB with DD', 'PR-WB with DD']
# line = ['b^--','g^-.']
cwp = os.getcwd()

fig, ax = plt.subplots(constrained_layout=True)

with open(
        cwp + '/configs/config.yaml'
) as f:
    config = yaml.load(f, Loader=yaml.loader.SafeLoader)

for l in range(len(configuration)):

    dataname = cwp + '/data/spatial/' + 'TX_RSRPdiff_RX' + configuration[l][0] \
               + '_sampleRate' + configuration[l][1]  + configuration[l][2]  \
               + configuration[l][3]  + configuration[l][4]  + '.npz'

    # dataname = cwp + '/data/spatial/'+codebook[l] +'_RX' + str(config['TrainRX'])+ '.npz'
    data = np.load(dataname)
    rsrp_diff = data['rsrp_diff']

    count, bins_count = np.histogram(rsrp_diff, bins=60)

    # finding the PDF of the histogram using count values
    pdf = count / sum(count)

    # using numpy np.cumsum to calculate the CDF
    # We can also find using the PDF values by looping and adding
    cdf = np.cumsum(pdf)

    # plot the cdf
    # ax.plot(bins_count[1:], pdf, color="red", label="PDF")
    ax.plot(bins_count[1:], cdf, line[l], label= label[l])

ax.set_title("CDF of Top-1 beam prediction RSRP error")
ax.set_xlabel("RSRP prediction error (dB)")
# ax.set_xticks(np.arange(0,10,6))
ax.title.set_size(16)
ax.xaxis.label.set_size(16)
ax.tick_params(axis='both', which='major', labelsize=14)
# plt.yscale('log')
plt.legend(fontsize=14)
plt.grid()
plt.show()