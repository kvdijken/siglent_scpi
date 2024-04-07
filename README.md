# siglent_scpi

Use networked Siglent devices from a computer over SCPI over ethernet.

This package lets users control and read Siglent devices which are connected by ethernet through the SCPI protocol. Some highlevel functions are included, like collecting a waveform, performing its FFT, and calculating the total harmonic distortion.

With this package automated measurements can be done. Advantages of taking automated measurements are:

- repeatability
- timesaving
- automated logging

The next plot shows an example experiment which has been performed with the library, showing the total harmonic distortion of a demodulated FM signal in three different scenario's. These measurements in total take about 15 minutes, while doing them manually would take several hours.

![Alt text](resources/thd_vs_modfreq.png)


## SDS

Class to connect to a Siglent oscilloscope. Currently the class has only been tested on a SDS1202X-E oscilloscope. It connects to the oscilloscope using sockets over ethernet.


## SDG

Class to connect to a Siglent signal generator. Currently the class has only been tested on a SDG1032X signal generator. It connects to the signal generator using sockets over ethernet.


## Example program

The example program thd_ch1.py shows how to get a continuously updated display of the FFT spectrum of channel 1 while also calculating and showing the total harmonic distortion.

An example output of the sample program is showing the output for a 1kHz signal from a signal generator (gif, low quality).

![Alt text](resources/Screencast-from-2024-04-06-21-19-27.gif)


## Things to be done:

- eror checking
- implement selection of windowing methods in SDS.fft()
- in SDS class, get equal parameters for channel selection in different methods (now a mix of string and int). This will break API.




