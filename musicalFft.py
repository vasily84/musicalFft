import matplotlib.pyplot as plt
import numpy as np
import scipy.fft as fft
from scipy.io.wavfile import write as write_wav, read as read_wav
from scipy.signal.windows import hamming
from subprocess import Popen, PIPE
import pyaudio,os,sys

# число видеокадров в секунду. 
VIDEO_FPS = 5
OCTAVE_FROM = -1
OCTAVE_TO = 3

def changeFileExt(fileName,fileExt):
    """изменить расширение файла """
    base = os.path.splitext(fileName)[0]
    return base + fileExt

def find_startIndex_and_intervalLength_to_transposition(octavePower,image_freq,f0,f1):
    """ найти стартовый индекс и длину интервала для отображения-транспозиции на частотный
    интервал от f0 до f1. octavePower - показатель, насколько транспонируется октава, image_freq - 
    ряд частот, f0,f1 - границы чатотного интервала, куда транспонируем.
    возвращает (startIndex,intervalLength) - (стартовый индекс начала интервала, длина интервала)."""
    
    startIndex = len(image_freq)
    LenCount = 0

    for i in range(1,len(image_freq)):
        f = image_freq[i]*(0.5)**(octavePower)
        if f>=f0 and startIndex==len(image_freq):
            startIndex = i

        if f>=f0 and f<=f1:
            LenCount += 1

    return (startIndex,LenCount)

def note2freq(note_='a', octave=1, A_=440):
    """ возвращает частоту ноты с учетом октавы в Равномерно Темперированном Строе.
    note_ - строка с буквенным обозначением ноты, octave - номер от октавы, 
    т.е. 1 - первая октава, 2 - вторая октава, 0 -малая октава и т.д.
    A_ - эталонная частота настройки ноты Ля первой октавы ."""
    octave -= 1
    def conv_Ai(i):
        return A_*2**(i/12)

    note = note_.lower()

    if note=='a': return conv_Ai(octave*12)
    elif note=='a#': return conv_Ai(1+octave*12)
    elif note=='b': return conv_Ai(2+octave*12)
    elif note=='c': return conv_Ai(-9+octave*12)
    elif note=='c#': return conv_Ai(-8+octave*12)
    elif note=='d': return conv_Ai(-7+octave*12)
    elif note=='d#': return conv_Ai(-6+octave*12)
    elif note=='e': return conv_Ai(-5+octave*12)
    elif note=='f': return conv_Ai(-4+octave*12)
    elif note=='f#': return conv_Ai(-3+octave*12)
    elif note=='g': return conv_Ai(-2+octave*12)
    elif note=='g#': return conv_Ai(-1+octave*12)
    elif note==' ': return 0. # специальная 'нота' - отсутствие звука

    raise ValueError("unknown musical note {}".format(note_))


def write_wav_test(fileName):
    """ записать тест для wav файла"""
    fps = 44100 # cd quality
    time = np.linspace(0,1,fps) # время 1 секунда
    frame_raw = np.ndarray(shape=(fps),dtype = np.int16)
    Ampl = 1000
    notes_track = ['c','d','e','f','g','a','b']
    audio_frames = []
    for i in range(-2,2):
        for n in notes_track:
            f = note2freq(n,i)
            frame_raw[:] = Ampl*np.sin(2*np.pi*f*time)
            audio_frames.append(np.copy(frame_raw))

    write_wav(fileName,fps,np.block(audio_frames))
        

def wav2guitar_distortion(inFile,outFile=None, deepValue=0.5):
    """ преобразовать звук в файле таким образом, как будто он
    прошел через дисторшн примочку"""
    if outFile is None:
        outFile = changeFileExt(inFile,'_distortion.wav')

    fps,data_in = read_wav(inFile)
    Amax = int(deepValue*np.max(data_in))
    Amin = int(deepValue*np.min(data_in))
    data_out = np.clip(data_in,Amin,Amax) # clip sound amplitude
    write_wav(outFile,fps,data_out)
          

class CAudioSource():
    def __init__(self,fileName):
        self.fileName = fileName
        if self.fileName is None: # данные из микрофона
            self._openMicrophone()
            self.readData = self._readDataMicrophone
            self.close = self._closeMicrophone
            return 
        #else:
        self._openWave(self.fileName)
        self.readData = self._readDataWave
        self.close = lambda : None # just stuff for close()
        
        
    def __enter__(self):
        print("CAudioSource __enter__")
        return self

    def __exit__(self,exc_type, exc_value, traceback):
        print("CAudioSource __exit__")
        self.close()

    def _openWave(self,fileName):
        self.fps,self.wav_buf = read_wav(fileName)
        channels = self.wav_buf.ndim
        if channels>1: # do stereo->mono sound transform
            self.wav_buf = self.wav_buf.flatten()[::channels]

        self.blockLen = int(self.fps/VIDEO_FPS)
        self.blocksReaded = 0
        self.blocksToRead= int(len(self.wav_buf)/self.blockLen)
        

    def _openMicrophone(self):
        self.audio = pyaudio.PyAudio()
        param = self.audio.get_default_input_device_info()
        self.fps = int(param.get('defaultSampleRate'))
        self.blockLen = int(self.fps/VIDEO_FPS)    
        self.stream = self.audio.open(format=pyaudio.paInt16, channels=1,
                rate=self.fps, input=True,
                frames_per_buffer=self.blockLen)
                

    def _readDataWave(self):
        """вычитать блок данных из потока"""
        s = self.blocksReaded
        block_data = self.wav_buf[s*self.blockLen:(s+1)*self.blockLen]
        self.blocksReaded += 1
        return (block_data,self.blocksReaded<self.blocksToRead)
    
    def _readDataMicrophone(self):
        data_raw = self.stream.read(self.blockLen)
        npdata = np.frombuffer(data_raw,dtype=np.int16)
        return (npdata,True)
    
    def _closeMicrophone(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

class CPlotter():
    def __init__(self,fileName=None):
        self.fileName = fileName
        if self.fileName is not None:
            self.silentVideoFile = changeFileExt(fileName,'_temp.mp4')
            self.soundVideoFile = changeFileExt(fileName,'.mp4')

    def __enter__(self):
        print("CPlotter __enter__")
        return self

    def __exit__(self,exc_type, exc_value, traceback):
        print("CPlotter __exit__")
        self.release()

    def _replot_init(self,image_fft,image_freq,FFTLEN,PLOT_FULL_SPECTR):
        self.image_freq = np.copy(image_freq)
        (self.fig, self.ax) = plt.subplots() 

        if not PLOT_FULL_SPECTR:
            plt.xscale('log')
            plt.yscale('log')   

        if PLOT_FULL_SPECTR:
            (self.lineAllScale,) = self.ax.plot(self.image_freq,np.abs(image_fft))
        else:
            self.sub_Lines = {}
            self.sub_image_freq = {}
            self.sub_startIndex = {}
            self.sub_intervalLength = {}
            f0 = note2freq('b',0)
            f1 = note2freq('c',2)
            for o2p in range(OCTAVE_FROM,OCTAVE_TO):
                startIndex,intervalLength = find_startIndex_and_intervalLength_to_transposition(o2p,image_freq,f0,f1)
                self.sub_startIndex[o2p] = startIndex
                self.sub_intervalLength[o2p] = intervalLength
                self.sub_image_freq[o2p] = self.image_freq[startIndex:startIndex+intervalLength]
                self.sub_image_freq[o2p] = self.sub_image_freq[o2p]*(0.5)**o2p
                sub_image_fft = image_fft[startIndex:startIndex+intervalLength] 
                (self.sub_Lines[o2p],) = self.ax.plot(self.sub_image_freq[o2p],np.abs(sub_image_fft))
            
            notes_f = []
            notes = ['c','c#','d','d#','e','f','f#','g','g#','a','a#','b']
            # рисуем вертикальные линии, по центрам нот
            for n in notes:
                f = note2freq(n)
                self.ax.vlines(f,1e-4,1e5,color='gray',linestyle=':')
                notes_f.append(f)
            
            self.ax.set_xticks(notes_f)
            # Устанавливаем подписи тиков
            self.ax.set_xticklabels(notes)
            plt.minorticks_off()    
        plt.ylim(1e-4,1e5)
        plt.show(block=False)

        if self.fileName is not None:
            self.pipe = Popen(['ffmpeg','-nostats','-loglevel','16', '-y', '-f', 'image2pipe', '-vcodec', 'png', '-r', str(VIDEO_FPS), '-i', '-', '-vcodec', 'mpeg4', '-qscale', '5', '-r', str(VIDEO_FPS), self.silentVideoFile], stdin=PIPE,bufsize = 0)

    def replot(self,image_fft,image_freq,FFTLEN,PLOT_FULL_SPECTR = False):
        """ вывести на нотный образ на график/записать в файл."""
        try:
            self.framesCount += 1
        except AttributeError: 
            # инициализация
            self.framesCount = 0
            self._replot_init(image_fft,image_freq,FFTLEN,PLOT_FULL_SPECTR)

        if PLOT_FULL_SPECTR:
            self.lineAllScale.set_data(self.image_freq,np.abs(image_fft))                
        else:
            for o2p in range(OCTAVE_FROM,OCTAVE_TO):
                startIndex = self.sub_startIndex[o2p]
                intervalLength = self.sub_intervalLength[o2p]
                sub_image_fft = image_fft[startIndex:startIndex+intervalLength] 
                self.sub_Lines[o2p].set_data(self.sub_image_freq[o2p],np.abs(sub_image_fft))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        if self.fileName is not None:
            # speed optimization plt.savefig() hack 
            buf = self.fig.canvas.tostring_rgb()
            ncols, nrows = self.fig.canvas.get_width_height()
            rgb_image = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
            plt.imsave(self.pipe.stdin,rgb_image,format='png')
            # or you can simply use:
            # fig.self.fig.savefig(self.pipe.stdin,format='png')
            # but it slows by 3 times
        
    def release(self):
        plt.close('all')
        if self.fileName is None: # nothing to release, just return 
            return 

        self.pipe.stdin.close()
        self.pipe.wait()
        # добавляем звук к видео
        cmd = "ffmpeg -nostats -loglevel 16 -y -i {} -i {} {}".format(self.fileName,self.silentVideoFile,self.soundVideoFile)
        os.system(cmd)
        os.remove(self.silentVideoFile)


def main():
    if len(sys.argv)>1:
        fileName = sys.argv[1]
    else:
        fileName = None
        
    """
    write_wav_test('wav_test.wav')
    wav2guitar_distortion('wav_test.wav')
    """
    fileName = 'music.wav'
    
    with CPlotter(fileName) as plot,CAudioSource(fileName) as sound:
        FFTLEN = fft.next_fast_len(sound.blockLen,True)
        image_freq = fft.rfftfreq(FFTLEN,1./sound.fps)
        repeat = True
        while repeat!=False:
            data,repeat = sound.readData()
            w = hamming(len(data),False)
            image_fft = fft.rfft(data*w,FFTLEN)
            plot.replot(image_fft,image_freq,FFTLEN)


if __name__=='__main__':
    main()