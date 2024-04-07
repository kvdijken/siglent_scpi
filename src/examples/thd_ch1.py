import numpy as np
import matplotlib.pyplot as plt

from siglent_scpi import SDS


plt.rcParams['toolbar'] = 'None'
plt.rcParams['axes.xmargin'] = 0

# channel to sample
ch = 0

# fundamental frequency from which to calculate the THD
f0 = 1000

# Max frequency of interest
max_f = 25e3

# ip address of oscilloscope
ip = "192.168.1.227"
sds = SDS(ip)


def plot(fig,ax):
    global lines

    def to_dBVrms(x):
        return 20*np.log10(x)

    drawn = 0
    
    # thd returns yf in V
    thd,xf,yf,bins = sds.thd(ch,f0,max_f,correct_peaks=True,min_level=-90)
    # transform to dBvrms for the display
    yf_dB = to_dBVrms(np.abs(yf))
    s0 = yf_dB[bins[0]]
    s1 = yf_dB[bins[1]]

    textstr = '\n'.join((r'$THD=%.2f$' % (thd, ) + '%',
    r'$s_0=%i dB_{V_{rms}}$' % (s0,),
    r'$s_1=%i dB_c$' % (int(s1-s0),),
    ))
    props = dict(boxstyle='round', facecolor='lightgrey', alpha=0.75)

    # place a text box in upper left in axes coords
    line = ax.text(0.4, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    lines.append(line)
    drawn = drawn + 1

    # harmonics
    for p in bins[1:]:
        line = ax.scatter(xf[p]/1000,yf_dB[p],marker='x',color='C2')
        lines.append(line)
        drawn = drawn + 1

    # fundamental frequency
    line = ax.scatter(xf[bins[0]]/1000,yf_dB[bins[0]],marker='o',color='C2')
    lines.append(line)
    drawn = drawn + 1

    # create plot, and remember the lines to remove them later
    line = ax.plot(xf/1000,yf_dB,color='C2',alpha=0.75,linewidth=1)[0]
    lines.append(line)
    drawn = drawn + 1

    # remove oldest of the previous lines (if any)
    while len(lines)>drawn:
        line = lines[0]
        lines.remove(line)
        line.remove()

    fig.canvas.draw_idle()
    fig.canvas.start_event_loop(0.01)


quit = False

def on_close(event):
    global quit
    print("Quiting ...")
    quit = True
    

fig, ax = plt.subplots()
fig.canvas.mpl_connect('close_event', on_close)
plt.title('FFT Channel 1')
plt.tight_layout(pad=4)
plt.xlabel('frequency (kHz)')
plt.ylabel(r'$dB_{V_{rms}}$')
plt.show(block=False)
plt.grid(True)
plt.ylim(-120,40)
lines = []
while not quit:
    plot(fig,ax)
