import struct
import errno
import os

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import flattop
import logging

from .siglent import Siglent, default_port



logger = logging.getLogger(__name__)

# constants for the SCPI commands
CH1 = 'C1'
CH2 = 'C2'


### Timebase lookup
tbase_lookup = np.array([200e-12,
                500e-12,
                1e-9,
                2e-9,
                5e-9,
                10e-9,
                20e-9,
                50e-9,
                100e-9,
                200e-9,
                500e-9,
                1e-6,
                2e-6,
                5e-6,
                10e-6,
                20e-6,
                50e-6,
                100e-6,
                200e-6,
                500e-6,
                1e-3,
                2e-3,
                5e-3,
                10e-3,
                20e-3,
                50e-3,
                100e-3,
                200e-3,
                500e-3,
                1,
                2,
                5,
                10,
                20,
                50,
                100])
_tbase_lookup = np.array(['200PS',
                '500PS',
                '1NS',
                '2NS',
                '5NS',
                '10NS',
                '20NS',
                '50NS',
                '100NS',
                '200NS',
                '500NS',
                '1US',
                '2US',
                '5US',
                '10US',
                '20US',
                '50US',
                '100US',
                '200US',
                '500US',
                '1MS',
                '2MS',
                '5MS',
                '10MS',
                '20MS',
                '50MS',
                '100MS',
                '200MS',
                '500MS',
                '1S',
                '2S',
                '5S',
                '10S',
                '20S',
                '50S',
                '100S'])
sq2 = np.sqrt(2)


# _bin is a synonym for _getnearpos
# _getnearpos finds the closest value in an array
def _bin(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx



class SDS(Siglent):

    _c = 1

    dislocations = 0
    peaks = 0


    def __init__(self, ip, port=default_port):
        '''
        Constructs an instance of the SDS class and connects
        to the oscilloscope using a ethernet socket to the 
        given ip address on the given port.

        Parameters:

        ip : string, ip address of the oscilloscope to connect to.
        port : int, port on which the oscilloscope is listening (default 5025).
        '''
        super().__init__()
        self._id = 'SDS'
        self.connect(ip,port)



    # This translates the display value (-127..128) to a voltage.
    def _toV(self,x,vgain,voffset,probe_att):
        return (x / 128 * vgain * 5 - voffset) * probe_att



    #
    def _to_dBVrms(self,x):
        return 20*np.log10(x)



    # Gets the wave for the channel
    # Returns (V,Tmax)
    def _getwave(self, channel):
        channel = channel + 1
        cmd = f'C{channel}:WF? DAT2'
        response = super().query(cmd,size=10000_000)

        # find the b'#9' marker
        _nine = response.index(b'#9')
        # the wave data starts at 11 bytes after b'#9' and ends with b'\n\n'
        wave = response[_nine+11:-2]
        w = np.frombuffer(wave,dtype=np.int8)

        # Get the waveform descriptor
        cmd = f'C{channel}:WF? DESC'
        response = super().query(cmd, size=100_000)
        desc = response[11+11:]
        vgain, = struct.unpack('f', desc[156:160])
        voffset, = struct.unpack('f', desc[160:164])
        probe_att, = struct.unpack('f', desc[328:332])
        _tb = struct.unpack('h', desc[324:326])[0]
        # sometimes the SDS1202X-E returns garbage in the waveform descriptor
        if _tb>=0 and _tb<tbase_lookup.size:
            timebase = tbase_lookup[_tb]
        else:
            return None, None

        v = self._toV(w,vgain,voffset,probe_att)
        return v, timebase*14



    # Gets the wave for the channel
    # Returns (t,V)
    # channel = 0 | 1
    def getwave(self, channel):
        '''
        Obtains the current waveform from the oscilloscope on the given channel.

        Parameters:

        channel : the channel for which to obtain the waveform from (0 or 1).

        Returns:
        t : numpy array with sample time, starting at 0 (in seconds)
        V : numpy array with voltage on time t (in V).
        '''

        y, Tmax = self._getwave(channel)
        t = np.linspace(0,Tmax,y.size,endpoint=False)
        return t, y


    # returns f, v (dB)
    # channel = 0 | 1
    # output = 'dBVrms' | 'dBV' | 'V'
    def fft(self, channel, max_f=None, output='dBVrms'):
        '''
        Calculates the Fast Fourier Transform of a waveform obtained from the oscilloscope.
        No error checking whatsoever is done.

        Parameters:

        channel : the channel to obtain the waveform for, either 0 (for channel 1)
                  or 1 (for channel 2).
        max_f : maximum frequency (in Hz) of the specturm to return. 
                If None, the complete spectrum will be returned.
        output : Determines in what format the result yf of the fft will be returned.
                 'dBVrms' = output yf in dBVrms
                 'dBV' = output yf in dBV
                 'V' = output yf in V

        Returns:

        xf : frequency bins [0..max_f]
        yf : magnitude bins [0..max_f]
        '''

        y, Tmax = self._getwave(channel)
        if y is None:
            return
        if Tmax is None:
            return
        
        N = y.size

        # samplerate (samples/second)
        samplerate = N / Tmax
        samplespacing = 1/samplerate
        xf = rfftfreq(N,samplespacing)

        # perform fft, returns yf in V
        yf = np.abs(rfft(y,norm='forward'))

        if output == 'dBVrms':
            yf = self._to_dBVrms(yf)
        elif output == 'dBV':
            yf = self._to_dBV(yf)

        # Return results up the the highest frequency of interest.
        if max_f is None:
            i = len(xf) - 1
        else:
            # Find the index of the highest frequency of interest
            # This highest frequency of interest will be included
            # in the returned results.
            i = np.nonzero(xf<=max_f)[0][-1]
            if xf[i] < max_f:
                # max_f is not included in xf[:i+1], include it, if possible.
                i = i + 1
                if i >= len(xf):
                    # prevent index out of range error
                    i = len(xf) - 1

        return xf[:i+1], yf[:i+1]
    

    def peakbin(self,yf,bin) -> int:
        m = bin
        # search up
        i = bin + 1
        while (i < len(yf)) and (yf[i] >= yf[m]):
            m = i
            i = i + 1
        # search down
        i = bin-1
        while (i>=0) and (yf[i] >= yf[m]):
            m = i
            i = i - 1
        return m


    # returns (thd,xf,yf,bins)
    # note that the returned yf is not in dB
    # thd in %
    # xf: frequency bins
    # yf: voltage in V
    # bins: indices in xf for harmonics
    #
    # channel = 0 | 1
    def thd(self, channel, f0, max_f=None,correct_peaks=False,min_level=None):
        """
        Calculates THD (total harmonic distortion, in %) of a waveform
        obtained from the oscilloscope.

        No error checking whatsoever is done.

        Parameters:

        channel : the channel to obtain the waveform for, either 0 (for channel 1)
                  or 1 (for channel 2).
        f0 : fundamental frequency (in Hz). The THD is calculated for this frequency..
        max_f : maximum frequency (in Hz) to consider. If None, the complete spectrum
                is included in the THD calculation.
        correct_peaks : If correct_peaks == True will correct harmonics bins if
                        directly neighbouring bins have higher signal level (default=False).
        min_level : minimum signal level (in dBV) for harmonics to be included in the
                    THD calculation (default=None). If min_level == None all harmonics bins
                    will be included, also if the signal appears to be below the noise level.

        Returns:

        thd : total harmonic distortion (in %)
        xf : array of frequencies of fft bins (in Hz), shortened to max_f if not None
        yf : array of signal level of fft bins (in V), same size as xf
        bins : array of harmonics bins (index into xf and yf).
        """

        xf, yf = self.fft(channel, max_f, output='V')

        # Harmonic frequencies within [f0:xf[-1]]
        freqs = np.arange(f0,xf[-1]+1,f0)

        # Get the bins for all harmonics
        bins = [_bin(xf,f) for f in freqs]

        if correct_peaks:
            # correct bins for the exact location of the nearby peak
            bins = [self.peakbin(yf,p) for p in bins]

        if min_level is not None:
            min = 10**(min_level/20) # from dBV to V
            # only take into regard bins for for yf>min
            # always include the fundamental
            bins = [i for i in bins if i==0 or yf[i]>min]

        # Calculate THD in %
        vsq = [yf[p]**2 for p in bins[1:]]
        vsq_sum = sum(vsq)
        thd = 100 * np.sqrt(vsq_sum) / yf[bins[0]]

        # calculate strength of strongest harmonic, assuming:
        s0 = self._to_dBVrms(yf[bins[0]])

        # debug info
        logger.info(f'f0 = {freqs[0]}, s0 = {np.round(s0,1)} dBvrms')
        n = len(bins)-1
        logger.info(f'#harmonics = {n}')
        for j in range(n):
            logger.info(f'f{j+1} = {freqs[j+1]}, s{j+1} = {np.round(self._to_dBVrms(yf[bins[j+1]]),2)} dBVrms')

        return thd,xf,yf,bins


    # Sets the oscilloscopes timebase
    # tb is an index in tbase_lookup
    def setTimebase(self,tb):
        '''
        Set the osciloscope timebase.

        Parameters:

        tb : index into tbase_lookup for the time / horizontal division.

        Returns:

        None
        '''
        cmd = 'TDIV ' + _tbase_lookup[tb]
        self.send(cmd)


    # Set the timebase to the first value larger than secs_per_div
    def setTimebaseAtLeast(self,secs_per_div):
        '''
        Set the oscilloscopes timebase to secs_per_div or the next 
        available larger timebase.

        Parameters:

        secs_per_div : minimal timebase per horizontal division (in seconds)

        Returns:

        None
        '''

        # Find first timebase in tbase_lookup which is larger than 'secs'
        tb = np.where(tbase_lookup > secs_per_div)[0][0]
        self.setTimebase(tb)


    # Returns the number of horizontal divisions
    def divisions(self):
        '''
        Returns the number of divisions for the oscilloscope.

        Parameters:

        None

        Returns:
        The number of divisions for the oscilloscope, 14.
        '''

        return 14


    # Returns the settings for ch as a dictionary
    # 'att' = attenuation (1 | 10 | 100 | ...)
    # 'bw' = bandwidth limited (True | False)
    # 'cpl' = coupling
    # 'offs' = offset (V)
    # 'skew' = skew (s)
    # 'trace' = trace ('ON' | 'OFF')
    # 'unit' = unit ()
    # 'vdiv' = volts per division (V)
    # 'invert' = inverted (True | False)
    #
    # ch = 'C1' | 'C2'
    def channelSettings(self,ch):
        '''
        Returns the settings for a channel on the oscilloscope.
        This is not a very quick method to return.

        Parameters:

        ch : 'C1' or 'C2', channel to obtain the settings for.

        Returns:

        Settings for the channel ch as a dictionary.

        Keys:
        'att' = attenuation (1 | 10 | 100 | ...)
        'bw' = bandwidth limited (True | False)
        'cpl' = coupling ('AC' or 'DC' or 'GND')
        'offs' = offset (V)
        'skew' = skew (s)
        'trace' = trace ('ON' | 'OFF')
        'unit' = unit ('V' or 'A')
        'vdiv' = volts per division (V)
        'invert' = inverted (True | False)
        '''

        # attenuation
        cmd = f'{ch}:ATTENUATION?'
        response = self.query(cmd).decode('ascii')
        # Returns: 'C1:ATTN 10\n'
        att = int(response[8:-1])

        # bandwidth limit
        cmd = f'{ch}:BANDWIDTH_LIMIT?'
        response = self.query(cmd).decode('ascii')
        # Returns:
        # 'C1:BWL ON\n'
        # 'C1:BWL OFF\n'
        match response[7:-1]:
            case 'ON':
                bw = True
            case 'OFF':
                bw = False

        # coupling
        #<coupling>:={A1M,A50,D1M,D50,GND}
        #A — alternating current.
        #D — direct current.
        #1M — 1MΩ input impedance.
        #50 — 50Ω input impedance.
        cmd = f'{ch}:COUPLING?'
        response = self.query(cmd).decode('ascii')
        match response[7:-1]:
            case 'A1M':
                cpl = 'AC'
            case 'A50':
                cpl = 'AC'
            case 'D1M':
                cpl = 'DC'
            case 'D50':
                cpl = 'DC'
            case 'GND':
                cpl = 'GND'

        # offset
        cmd = f'{ch}:OFFSET?'
        response = self.query(cmd).decode('ascii')
        # Returns: 'C1:OFST 0.00E+00V\n'
        offs = float(response[8:-2])

        # skew
        cmd = f'{ch}:SKEW?'
        response = self.query(cmd).decode('ascii')
        # Returns: 'C1:SKEW 0.00E+00S\n'
        skew = float(response[8:-2])

        # trace
        cmd = f'{ch}:TRACE?'
        response = self.query(cmd).decode('ascii')
        # Returns:
        # 'C1:TRA ON\n'
        # 'C1:TRA OFF\n'
        trace =  response[7:-1]

        # unit
        cmd = f'{ch}:UNIT?'
        response = self.query(cmd).decode('ascii')
        # Returns:
        # 'C1:UNIT V\n'
        # 'C1:UNIT A\n'
        unit = response[8:-1]

        # Volts per division
        cmd = f'{ch}:VDIV?'
        response = self.query(cmd).decode('ascii')
        # Returns 'C1:VDIV 2.00E-01V\n'
        vdiv = float(response[8:-2])

        # inverted
        cmd = f'{ch}:INVS?'
        response = self.query(cmd).decode('ascii')
        # Returns:
        # 'C1:INVS ON\n'
        # 'C1:INVS OFF\n'
        match response[8:-1]:
            case 'ON':
                invert = True
            case 'OFF':
                invert = False

        return {'att':att, 
                'bw':bw, 
                'cpl':cpl, 
                'offs':offs, 
                'skew':skew, 
                'trace': trace, 
                'unit':unit,
                'vdiv':vdiv,
                'invert':invert}
    

    def timebaseSettings(self):
        '''
        Returns the timebase settings for the oscilloscope.

        Parameters:

        None

        Returns:

        Returns the horizontal timebase in seconds per division.
        '''

        cmd = f'TIME_DIV?'
        response = self.query(cmd).decode('ascii')

        # for a 10ms timebase "TDIV 1.00E-02S\n" is returned.
        return float(response[5:-2])
    

    def memorySettings(self):
        '''
        Returns the memory size setting for the oscilloscope.

        Parameters:

        None

        Returns:

        Returns the memory size as a string:
        'MSIZ 14K' = 14_000
        'MSIZ 140K' = 140_000
        'MSIZ 1.4M' = 1_400_000
        'MSIZ 14M' = 14_000_000
        '''

        cmd = 'MEMORY_SIZE?'
        response = self.query(cmd).decode('ascii')
        # returns:
        # 'MSIZ 14K\n'
        # 'MSIZ 140K\n'
        # 'MSIZ 1.4M\n'
        # 'MSIZ 14M\n'
        match response[5:-1]:
            case '14K':
                mem = 14_000
            case '140K':
                mem = 140_000
            case '1.4M':
                mem = 1_400_000
            case '14M':
                mem = 14_000_000
            case _:
                mem = None
        return mem


    def acquireSettings(self):
        '''
        Returns the acquisition settings for the oscilloscope.

        Parameters:

        None

        Returns:

        Returns a string representing the sampling method and a float 
        for the samplerate in samples per second.

        Sampling string:
        'SAMPLING' = normal
        'PEAK_DETECT' = peak detection
        'AVERAGE,16' = averaging, no details
        'HIGH_RES' = high resolution, no details
        '''

        cmd = 'ACQW?'
        response = self.query(cmd).decode('ascii')
        # returns:
        # 'ACQW SAMPLING\n'
        # 'ACQW PEAK_DETECT\n'
        # 'ACQW AVERAGE,16\n'
        # 'ACQW HIGH_RES\n'
        acq = response[5:-1]

#        cmd = 'AVERAGE_ACQUIRE?'
#        response = self.query(cmd).decode('ascii')
        # returns: 'AVGA 16\n'
#        avg = int(response[5:-1])

        # sample rate
        cmd = 'SARA?'
        # Returns: 'SARA 1.00E+09Sa/s\n'
        response = self.query(cmd).decode('ascii')
        sara = float(response[5:-5])

        # For now don't bother decoding this
        return acq, sara
    

    # Collect the settings from the SDS1202X-E
    # output:
    # 'tuple': returns settings as tuple
    # 'table': returns settings as a markdown table (use Ctrl-Shift-V to paste into Obsidian)
    # 'text': returns settings as printable text
    def settings(self,output='tuple'):
        '''
        Returns the settings (configuration) of the oscilloscope.
        The format to be returned can be given by the parameter output.
        Configuration details include channel settings (for both channels),
        timebase settings, memory size setting and acquisition settings.

        Parameters:

        output : determines the format of the result
                 'tuple' = return the result as a tuple (tb, ch1, ch2, mem, acq) 
                 'table' = return the result as a tale in markdown format
                 'text' = return the result as a multiline string which can be 
                          sent to the console or a file.
        '''
        
        def tupleToTable(tb, ch1, ch2, mem, acq):
            txt = ''

            def add(line):
                nonlocal txt
                if txt == '':
                    txt = line
                else:
                    txt = txt + '\n\r' + line

            def addChannel(ch,t):
                add(f"|{t}|Attenuation|{ch['att']}|")
                add(f"||Bandwidth limited|{ch['bw']}|")
                add(f"||Coupling|{ch['cpl']}|")
                add(f"||Offset (V)|{ch['offs']}|")
                add(f"||Skew (s)|{ch['skew']}|")
                add(f"||Trace|{ch['trace']}|")
                add(f"||Unit|{ch['unit']}|")
                add(f"||V/div|{ch['vdiv']}|")
                add(f"||Inverted|{ch['invert']}|")

            add('|Channel|Setting|Value|')
            add('|-|-|-|')
            add(f'||Timebase (s/div)|{tb}|')
            add(f'||Memory Depth|{mem}|')
            add(f'||Acquire|{acq[0]}|')
            add(f'||Samplerate (Sa/s)|{acq[1]}|')
            add('||||')
            addChannel(ch1,'CH1')
            add('||||')
            addChannel(ch2,'CH2')
            return txt
        
        def tupleToText(tb, ch1, ch2, mem, acq):
            txt = ''

            def add(line):
                nonlocal txt
                if txt == '':
                    txt = line
                else:
                    txt = txt + '\n\r' + line

            def addChannel(ch,t):
                add(f'Channel {t}')
                add('---------')
                add(f"Attenuation = {ch['att']}x")
                add(f"Bandwidth limited = {ch['bw']}")
                add(f"Coupling = {ch['cpl']}")
                add(f"Offset = {ch['offs']} V")
                add(f"Skew = {ch['skew']} s")
                add(f"Trace = {ch['trace']}")
                add(f"Unit = {ch['unit']}")
                add(f"V/div = {ch['vdiv']}")
                add(f"Inverted = {ch['invert']}")


            add(f'Timebase = {tb} s/div')
            add(f'Memory Depth = {mem} points')
            add(f'Acquire = {acq[0]}')
            add(f'Samplerate = {acq[1]} Sa/s')
            add('')
            addChannel(ch1,'1')
            add('')
            addChannel(ch1,'2')
            return txt
        
        ch1 = self.channelSettings(CH1)
        ch2 = self.channelSettings(CH2)
        tb = self.timebaseSettings()
        mem = self.memorySettings()
        acq = self.acquireSettings()
        match output:
            case 'tuple':
                return tb, ch1, ch2, mem, acq[0], acq[1]
            case 'table':
                return tupleToTable(tb, ch1, ch2, mem, acq)
            case 'text':
                return tupleToText(tb, ch1, ch2, mem, acq)
    

    def stop(self):
        '''
        Stops the acquisition of the oscilloscope. Note that a
        corresponding 'run' (or 'start') method, to start acquisition
        of the oscilloscope does not exist yet.

        Parameters:

        None

        Returns:

        None
        '''

        cmd = 'STOP'
        self.send(cmd)


    # This ('ARM') does not work
    # Not found a way to do it yet.
    def run(self):
        pass
