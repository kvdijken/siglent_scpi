import socket
import time
import os
import logging

logger = logging.getLogger(__name__)

default_port = 5025

class Siglent:

    _sock = None
    _ip = None
    _port = default_port

    # Create an TCP socket
    def _SocketConnect(self):
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except socket.error:
            logger.info ('Failed to create socket.')
            return None
            
        try:
            self._sock.connect((self._ip, self._port))
            self._sock.setblocking(False)
        except socket.error:
            logger.info ('Failed to connect to ip ' + self._ip)
            self._sock.close()
            return None
        return self._sock


    # Close the socket
    def _SocketClose(self):
        self._sock.close()
        time.sleep(1)


    # Send message to the device
    def _SocketSend(self, cmd, size):
        try :
            self._sock.sendall(cmd + b'\n')
        except socket.error as error:
            #Send failed
            logger.info ('Send failed')
            logger.info(os.strerror(error.errno))


    def _SocketQuery(self, cmd, size):
        try :
            self._sock.sendall(cmd + b'\n')
        except socket.error as error:
            #Send failed
            logger.info('Send failed')
            logger.info(os.strerror(error.errno))

        data = None    
        while True:
            try:
                time.sleep(0.1)
                if data == None:
                    data = self._sock.recv(size)
                else:
                    data += self._sock.recv(size)
            except BlockingIOError:
                if data == None:
                    pass
                else:
                    return data


    #
    def disconnect(self):
        if self._sock is not None:
            self._SocketClose()


    #
    def connect(self, ip, port=default_port):
        if self._sock is not None:
            self.disconnect()
        self._ip = ip
        _port = port
        return self._SocketConnect()


    #
    def send(self, cmd: str, size=8000):
        logger.info(f'SEND {self._id}: {cmd}')
        b_cmd = bytes(cmd,'ascii')
        self._SocketSend(b_cmd,size)
    

    # 
    def query(self, cmd: str, size=8000):
        logger.info(f'QUERY {self._id}: {cmd}')
        b_cmd = bytes(cmd,'ascii')
        return self._SocketQuery(b_cmd,size)
    

    #
    def __init__(self) -> None:
        pass


    #
    def __del__(self):
        if self._sock is not None:
            self.disconnect()


