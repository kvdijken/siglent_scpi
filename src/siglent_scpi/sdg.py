import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import flattop
import logging

from .siglent import Siglent, default_port


logger = logging.getLogger(__name__)


CHANNEL = ['C1','C2']

class SDG(Siglent):


    def __init__(self, ip, port=default_port):
        super().__init__()
        self._id = 'SDG'
        self.connect(ip,port)


    def query(self, cmd: str, size=8000):
        response = super().query(cmd,size)
        logger.info(f'RESPONSE: {response}')
        return response
    

    def MDWV(self, ch, type=None, parameter=None, value=None):
        if type is None:
            cmd = CHANNEL[ch] + ':MDWV ' + parameter + ',' + value
        else:
            cmd = CHANNEL[ch] + ':MDWV ' + type
        return self.send(cmd)


