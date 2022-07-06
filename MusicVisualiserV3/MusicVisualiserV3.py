DEBUG = False

def validateArgs(samplerate):
    if args.fps <= 0 or samplerate % args.fps != 0:
        print("Error: fps of {0} is invalid".format(args.fps))
        sys.exit()

    if args.size > 90 or args.size < 0:
        print("Error: size of {0} is invalid as is it is over 90 or under 0".format(args.size))
        sys.exit()

    if args.offset < 0 or args.offset > (samplerate - (args.size * args.size)):
        print("Error: offset of {0} is invalid".format(args.offset))
        sys.exit()

    if args.scalefactor < 1:
        print("Error: scalefactor of {0} is invalid".format(args.scalefactor))
        sys.exit()

    if args.width < 0 or samplerate % args.width != 0:
        print("Error: windowwidth of {0} is invalid".format(args.width))
        sys.exit()

    if args.size != 90 and args.spiral:
        print("Error: size cannot be changed when using -s or -j")
        sys.exit()

def loadaudio(path, tempfolderpath):
    #get path
    inpath = os.path.abspath(path)

    name = os.path.splitext(os.path.basename(inpath))[0]
    extension = os.path.splitext(os.path.basename(inpath))[1]

    if extension == "":
        inpath = inpath + ".wav"

    print("CONVERTING AUDIO TO WAV \n")
    os.system("ffmpeg -y -i \"{}\" -ar 44100 -ac 2 \"{}\"{}".format(inpath, tempfolderpath.format(".wav"), " > nul 2>&1"))
    convpath = os.path.abspath(tempfolderpath.format(".wav"))

    #attempt load audio
    try:
        samplerate, memaudio = read(convpath)
    except:
        print("Could not load audio. Could not find file {0} or converted file {1}".format(inpath, convpath))
        sys.exit()

    try:
        assert memaudio.shape[1] == 2
    except:
        try:
            print("Could not load audio. Audio has {0} channels, not 2".format(memaudio.shape[1]))
        except:
            print("Could not load audio. Audio has 1 channel, not 2")
        sys.exit()

    if samplerate % args.fps != 0:
        print("Sample rate of audio must be multiple of fps")
        sys.exit()

    samplesperframe = samplerate // args.fps

    #pad audio to round number of frames
    length = len(memaudio)
    memaudio = np.pad(memaudio, ((0, (((length // samplesperframe) + 1) * samplesperframe) - length), (0, 0)))
    length = len(memaudio)

    #save audio to disk as opposed to memory
    audio = np.memmap(tempfolderpath.format("_AUDIO.memmap"), dtype=np.int16, mode="w+", shape=memaudio.shape)
    audio[:] = memaudio[:]
    audio.flush()
    del memaudio

    return samplerate, samplesperframe, length, audio, inpath

def removePercentile(a, b, c, p):
    Q1 = np.percentile(a, p)
    Q2 = np.percentile(b, p)
    Q3 = np.percentile(c, p)
    a[a < Q1] = Q1
    b[b < Q2] = Q2
    c[c < Q3] = Q3
    a -= Q1
    b -= Q2
    c -= Q3

def getWindow(i, audio, samplesperframe, lengthinframes):
    #find start of window
    startindex = max(((i - (args.fps // 2)) * samplesperframe), 0)
    #how much padding in front
    paddingfront = 0 if (i - (args.fps // 2)) >= 0 else ((args.fps // 2) - i) * samplesperframe

    #find end of window
    endindex = min((i + (args.fps // 2)) * samplesperframe, lengthinframes * samplesperframe)
    #how much end padding
    paddingback = 0 if (i + (args.fps // 2)) <= lengthinframes else ((i + (args.fps // 2)) - lengthinframes) * samplesperframe

    #get window
    window1 = np.multiply(np.pad(audio[startindex:endindex, 0], (paddingfront, paddingback)), rollingwindow)
    window2 = np.multiply(np.pad(audio[startindex:endindex, 1], (paddingfront, paddingback)), rollingwindow)
    #third channel is average
    window3 = (window1 + window2) / 2
    return window1, window2, window3

def sqrtabsfft(window1, window2, window3):
    #start and end of output
    startindex = args.offset
    endindex = args.offset + args.area

    #compute the fft, then abs that, then sqrt
    transformed1 = np.sqrt(np.absolute(np.fft.fft(window1)[startindex:endindex]).astype(np.float64))
    transformed2 = np.sqrt(np.absolute(np.fft.fft(window2)[startindex:endindex]).astype(np.float64))
    transformed3 = np.sqrt(np.absolute(np.fft.fft(window3)[startindex:endindex]).astype(np.float64))
    return transformed1, transformed2, transformed3

def fft(window1, window2, window3):
    #start and end of output
    startindex = args.offset
    endindex = args.offset + args.area

    #compute the fft, then abs that
    transformed1 = np.absolute(np.fft.fft(window1)[startindex:endindex]).astype(np.float64)
    transformed2 = np.absolute(np.fft.fft(window2)[startindex:endindex]).astype(np.float64)
    transformed3 = np.absolute(np.fft.fft(window3)[startindex:endindex]).astype(np.float64)
    return transformed1, transformed2, transformed3

def normaliseFrequency(transformed1, transformed2, transformed3):
    #start and end of output
    startindex = args.offset
    endindex = args.offset + args.area

    #normalise intensity using ISO226:2003 human loudness curves
    gradient = data.ISO226[startindex:endindex]
    #rough (and i mean rough) approximation of human loudness curve. assumes bass will be louder/heard less
    #gradient = np.log1p(np.arange(startindex, endindex)) / np.log1p(args.area)

    transformed1 *= gradient
    transformed2 *= gradient
    transformed3 *= gradient

def generateFrame(transformed1, transformed2, transformed3, frame):
    #assign computed data to frame channels as required
    frame[:, :, c1] = transformed1.reshape((args.size, args.size)) #channel1
    frame[:, :, c2] = transformed2.reshape((args.size, args.size)) #channel2
    frame[:, :, c3] = transformed3.reshape((args.size, args.size)) #fixed

def process(audio, videoL, transform, usememmap, samplerate):
    #create file on disk for frames to be stored
    if usememmap:
        framemap = np.memmap(tempfolderpath.format("_FRAMES.memmap"), dtype=np.uint8, mode="w+", shape=(lengthinframes, args.size, args.size, 3))
    else:
        framemap = None

    #preassign arrays
    window1, window2, window3 = np.zeros(samplerate, dtype=np.float64), np.zeros(samplerate, dtype=np.float64), np.zeros(samplerate, dtype=np.float64)

    ##smoothing array
    #datamax = 1
    #width = 0.2
    #maxarray = np.full(int(args.fps * width), 1, np.float64)
    samplesperframe = samplerate // args.fps

    #preassign vars
    frame = np.zeros((args.size, args.size, 3), np.uint8)
    totaltime = ffttime = transformtime = 0
    for i in range(0, lengthinframes):
        if i % 30 == 0:
            #save changes to disk (keeps memory low)
            if usememmap:
                framemap.flush()

            #progress
            print("PROCESSING {0:5.2f}% | {1:^2}ms/frame | {2:>3}fps | fft: {3:>2}ms | ETA: {4}min".format(
                ((i / lengthinframes) * 100), 
                  int(totaltime*1000),
                  int(1/max(totaltime, 1.1/1000)),
                  int(ffttime*1000),
                  round(((lengthinframes-i) * totaltime)/60, 1)),
                  end="\r")

        framestarttime = time.time()

        #get windows of audio data
        window1, window2, window3 = getWindow(i, audio, samplesperframe, lengthinframes)
        assert window1.size == window2.size == window3.size == samplerate

        s = time.time()
        #fft data
        transformed1, transformed2, transformed3 = fft(window1, window2, window3)
        ffttime = time.time() - s

        #normalisation/dsp
        removePercentile(transformed1, transformed2, transformed3, 2.5)
        normaliseFrequency(transformed1, transformed2, transformed3)


        #smooth data so frames dont flash (really annoying)
        #maxarray = np.roll(maxarray, -1)
        #maxarray[-1] = max(transformed1.max(), transformed2.max(), transformed3.max())
        datamax = max(transformed1.max(), transformed2.max(), transformed3.max())#maxarray.max()

        #resize data to fit in uint8
        transformed1 *= (255 / datamax)
        transformed2 *= (255 / datamax)
        transformed3 *= (255 / datamax)

        #if frame is too bright, decrease it
        if np.sum(transformed3) > args.area * 256 * 0.25:
            transformed1 *= ((args.area * 256 * 0.25) / np.sum(transformed3))
            transformed2 *= ((args.area * 256 * 0.25) / np.sum(transformed3))
            transformed3 *= ((args.area * 256 * 0.25) / np.sum(transformed3))

        #convert to uint8
        transformed1, transformed2, transformed3 = transformed1.astype(np.uint8), transformed2.astype(np.uint8), transformed3.astype(np.uint8)

        #make bitmap image
        generateFrame(transformed1, transformed2, transformed3, frame)

        totaltime = time.time() - framestarttime

        #write frame to video/framesmap
        if args.lines or args.all:
            videoL.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if usememmap:
            framemap[i, :, :, :] = frame[:, :, :]
    else:
        #garbage collection
        if args.lines:
            videoL.release()
        audio._mmap.close()
        del audio
        del frame
        del transformed1
        del transformed2
        del transformed3
        return framemap

def postprocess(framemap, videoS, videoH):

    if args.all:
        ppSpiral(framemap, videoS)
        ppHilbert(framemap, videoH)
    elif args.spiral:
        ppSpiral(framemap, videoS)
    elif args.hilbert:
        ppHilbert(framemap, videoH)

def ppSpiral(framemap, videoS):
    print()
    print()
    transformtime = 0
    framemap.flush()

    transformedmap = np.memmap(tempfolderpath.format("_TRANSS.memmap"), dtype=np.uint8, mode="w+", shape=(lengthinframes, 90, 90, 3))
    transformedmap._mmap.close()

    i = 0

    while True:
        print("POST-PROCESSING {0:5.2f}%".format((i / lengthinframes) * 100), end ="\r")
        #horrifyingly janky way of avoiding memory leak in 3rd party gpu code :(
        if DEBUG:
            location = "E:/MusicVisualiserOutput/postprocessing.exe"
        else:
            location = "./postprocessing.exe"

        result = subprocess.run("\"{}\" {} \"{}\" \"{}\" {} {} {}".format(location, lengthinframes, tempfolderpath, tempfolderpath.format("_TRANSS.memmap"), i, 1, False), capture_output=True, text=True)

        if result.returncode == 0:
            break
        else:
            i = int(result.stdout.strip())

    print()
    print()
    transformedmap = np.memmap(tempfolderpath.format("_TRANSS.memmap"), dtype=np.uint8, mode="r+", shape=(lengthinframes, 90, 90, 3))
    for i in range(0, lengthinframes):
        if i % 30 == 0:
            #progress
            print("WRITING VIDEO {0:5.2f}% | {1:^2}ms/frame | {2:>3}fps | ETA: {3}min".format(
                ((i / lengthinframes) * 100), 
                    int(transformtime*1000),
                    int(1/max(transformtime, 1.1/1000)),
                    round(((lengthinframes-i) * transformtime)/60, 1)),
                    end="\r")

        s = time.time()

        #get result
        frame = transformedmap[i, :, :, :]
        transformtime = time.time() - s

        #write to video
        videoS.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    transformedmap._mmap.close()

def ppHilbert(framemap, videoH):
    print()
    print()
    transformtime = 0
    framemap.flush()

    transformedmap = np.memmap(tempfolderpath.format("_TRANSH.memmap"), dtype=np.uint8, mode="w+", shape=(lengthinframes, 64, 64, 3))
    transformedmap._mmap.close()

    i = 0

    while True:
        print("POST-PROCESSING {0:5.2f}%".format((i / lengthinframes) * 100), end ="\r")
        #horrifyingly janky way of avoiding memory leak in 3rd party gpu code :(
        if DEBUG:
            location = "E:/MusicVisualiserOutput/postprocessing.exe"
        else:
            location = "./postprocessing.exe"

        result = subprocess.run("\"{}\" {} \"{}\" \"{}\" {} {} {}".format(location, lengthinframes, tempfolderpath, tempfolderpath.format("_TRANSH.memmap"), i, 0, str(args.all)), capture_output=True, text=True)

        if result.returncode == 0:
            break
        else:
            i = int(result.stdout.strip())

    print()
    print()
    transformedmap = np.memmap(tempfolderpath.format("_TRANSH.memmap"), dtype=np.uint8, mode="r+", shape=(lengthinframes, 64, 64, 3))
    for i in range(0, lengthinframes):
        if i % 30 == 0:
            #progress
            print("WRITING VIDEO {0:5.2f}% | {1:^2}ms/frame | {2:>3}fps | ETA: {3}min".format(
                ((i / lengthinframes) * 100), 
                    int(transformtime*1000),
                    int(1/max(transformtime, 1.1/1000)),
                    round(((lengthinframes-i) * transformtime)/60, 1)),
                    end="\r")

        s = time.time()

        #get result
        frame = transformedmap[i, :, :, :]
        transformtime = time.time() - s

        #write to video
        videoH.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    transformedmap._mmap.close()


def main():
    #test if ffmpeg is installed
    ffmpeg = True
    if os.system("ffmpeg -version > nul 2>&1") != 0:
        ffmpeg = False
        print("FFMPEG is not installed, output will be low res and will not have audio")
        input()

    #parse arguments
    parser = argparse.ArgumentParser(description="Convert an audio file into a visual representation", prog="MusicVisualiserV3.0")
    parser.add_argument("filename", help="The location of an input audio file. Must include exactly two audio channels")
    parser.add_argument("-fps", "-f", nargs="?", type=int, default=60, 
                        help="The desired fps of the output video. Must perfectly divide the input samplerate")
    parser.add_argument("-size", "-s", nargs="?", type=int, default=90, 
                        help="The desired size of the output video. The output video will be a square of the given size in pixels. Must be <= 90. Cannot be changed if using -H, -S or -A")
    parser.add_argument("-offset", "-o", nargs="?", type=int, default=32, 
                        help="The frequency, in Hz, to offset the output by. Must be >=0 and <= samplerate - size^2")
    parser.add_argument("-scalefactor", "-k", nargs="?", type=int, default=5, 
                        help="The scale factor to increase the output size by. Must be >= 1. Values < 10 are recommended")
    parser.add_argument("-width", "-n", nargs="?", type=int, default=4410, 
                        help="The width of the window used to sample. Must perfectly divide samplerate")
    parser.add_argument("-spectrogram", "-i", nargs="?", type=bool, const=True, default=False,
                        help="Generate a spectrogram of the audio processed")
    group = parser.add_mutually_exclusive_group()
    group.title = "Transformations"
    group.add_argument("-lines", "-L", action="store_true",
                        help="Create output in lines form (default)")
    group.add_argument("-spiral", "-S", action="store_true",
                        help="Create output in spiral form, as opposed to lines")
    group.add_argument("-hilbert", "-H", action="store_true",
                        help="Create output in hilbert form, as opposed to lines. Produces a blocky output that better represents locality")
    group.add_argument("-all", "-A", action="store_true",
                        help="Create all outputs, lines, hilbert and spiral")
    group2 = parser.add_mutually_exclusive_group()
    group2.title = "Colour"
    group2.add_argument("-R", action="store_true",
                        help="Red is the fixed colour and thus will not appear in output\nProduces a purple-lime spectrum")
    group2.add_argument("-G", action="store_true",
                        help="Green is the fixed colour and thus will not appear in output\nProduces a orange-blue spectrum")
    group2.add_argument("-B", action="store_true",
                        help="Blue is the fixed colour and thus will not appear in output.\nProduces a red-green spectrum")
    parser.add_argument("-F", action="store_true",
                        help="Flips the orientation of the two colours")

    global args #sue me
    args = parser.parse_args()


    #setup values
    if args.all:
        args.size = 90
    elif args.hilbert:
       args.size = 64
    elif args.spiral:
        args.size = 90

    args.area = args.size ** 2

    outsize = args.size * args.scalefactor
    windowwidth = args.width
    usememmap = args.hilbert or args.spiral or args.spectrogram or args.all
    transform = args.hilbert or args.spiral or args.all

    args.B = not(args.R or args.G)
    global c1, c2, c3 #sue me
    if args.B:
        c1 = 1 if args.F else 0
        c2 = 0 if args.F else 1
        c3 = 2
    if args.R:
        c1 = 1 if args.F else 2
        c2 = 2 if args.F else 1
        c3 = 0
    if args.G:
        c1 = 2 if args.F else 0
        c2 = 0 if args.F else 2
        c3 = 1

    args.lines = not(args.hilbert or args.spiral)


    #manage directories
    audiofilename = os.path.basename(os.path.abspath(args.filename))
    outfilename = str(int(time.time()))

    outputfolder = os.path.dirname(os.path.abspath(args.filename)) + "\\" + os.path.splitext(os.path.basename(os.path.abspath(args.filename)))[0] + "\\"
    outputfilepath = outputfolder + "{0}\\" + outfilename + "{1}"
    global tempfolderpath
    tempfolderpath = outputfilepath.format("TEMP", "{0}")

    #create temp dir
    if not os.path.exists(os.path.join(outputfolder, "TEMP\\")):
        os.makedirs(os.path.join(outputfolder, "TEMP\\"))

    #create spect dir
    if args.spectrogram:
        if not os.path.exists(os.path.join(outputfolder, "SPECTROGRAM\\")):
            os.makedirs(os.path.join(outputfolder, "SPECTROGRAM\\"))

    #create vid dir
    if not os.path.exists(os.path.join(outputfolder, "VIDEO\\")):
        os.makedirs(os.path.join(outputfolder, "VIDEO\\"))

    #create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

    videoH = cv2.VideoWriter(outputfilepath.format("TEMP", "H.mp4"), fourcc, args.fps, (64, 64))
    videoS = cv2.VideoWriter(outputfilepath.format("TEMP", "S.mp4"), fourcc, args.fps, (90, 90))
    videoL = cv2.VideoWriter(outputfilepath.format("TEMP", "L.mp4"), fourcc, args.fps, (90, 90))

    #load audio
    samplerate, samplesperframe, length, audio, pathtoaudio = loadaudio(args.filename, tempfolderpath)
    validateArgs(samplerate)

    #analyse audio
    global lengthinframes
    lengthinframes = length // (samplerate // args.fps)
    lengthinseconds = lengthinframes / args.fps

    #create sample window
    paddingwidth = (samplerate // 2) - (windowwidth // 2)
    global rollingwindow
    rollingwindow = np.pad(signal.get_window('hann', windowwidth), (paddingwidth, paddingwidth))

    #process audio
    framemap = process(audio, videoL, transform, usememmap, samplerate)

    #transform video by specified method
    if transform:
        postprocess(framemap, videoS, videoH)

    #save spectrogram
    if args.spectrogram:
        print()
        print("\nSAVING SPECTROGRAM OF SIZE {0:2.1f}GB".format((framemap.size / (1024**3))), end="\r")

        img = Image.fromarray(np.rot90(framemap.reshape((lengthinframes, args.area, 3))))
        img.save(outputfilepath.format("SPECTROGRAM", ".png"))
        del img
        print("SAVED SPECTROGRAM                       \n")

    #done with frames
    if usememmap:
        framemap._mmap.close()
        del framemap


    usehwaccel = ("nvenc" in subprocess.getoutput('ffmpeg'))
    videoL.release()
    videoH.release()
    videoS.release()

    if args.spiral or args.all:
        #scale video
        os.system("ffmpeg -y -i \"{}\" -vf scale={}:{}:flags=neighbor -threads {} {}\"{}\"".format(tempfolderpath.format("S.mp4"), outsize, outsize, psutil.cpu_count() // 2, "-c:v h264_nvenc " if usehwaccel else "", tempfolderpath.format("S_2.mp4")))
        #add audio
        os.system("ffmpeg -y -i \"{}\" -i \"{}\" -map 0:v -map 1:a -c:v copy -shortest \"{}\"".format(tempfolderpath.format("S_2.mp4"), pathtoaudio, outputfilepath.format("VIDEO", "S.mp4")))
    if args.hilbert or args.all:
        #scale video
        os.system("ffmpeg -y -i \"{}\" -vf scale={}:{}:flags=neighbor -threads {} {}\"{}\"".format(tempfolderpath.format("H.mp4"), outsize, outsize, psutil.cpu_count() // 2, "-c:v h264_nvenc " if usehwaccel else "", tempfolderpath.format("H_2.mp4")))
        #add audio
        os.system("ffmpeg -y -i \"{}\" -i \"{}\" -map 0:v -map 1:a -c:v copy -shortest \"{}\"".format(tempfolderpath.format("H_2.mp4"), pathtoaudio, outputfilepath.format("VIDEO", "H.mp4")))
    if args.lines or args.all:
        #scale video
        os.system("ffmpeg -y -i \"{}\" -vf scale={}:{}:flags=neighbor -threads {} {}\"{}\"".format(tempfolderpath.format("L.mp4"), outsize, outsize, psutil.cpu_count() // 2, "-c:v h264_nvenc " if usehwaccel else "", tempfolderpath.format("L_2.mp4")))
        #add audio
        os.system("ffmpeg -y -i \"{}\" -i \"{}\" -map 0:v -map 1:a -c:v copy -shortest \"{}\"".format(tempfolderpath.format("L_2.mp4"), pathtoaudio, outputfilepath.format("VIDEO", "L.mp4")))

    ##scale video
    #os.system("ffmpeg -y -i \"{}\" -vf scale={}:{}:flags=neighbor -threads {} {}\"{}\"".format(tempfolderpath.format(".mp4"), outsize, outsize, psutil.cpu_count() // 2, "-c:v h264_nvenc " if usehwaccel else "", tempfolderpath.format("_2.mp4")))
    ##add audio
    #os.system("ffmpeg -y -i \"{}\" -i \"{}\" -map 0:v -map 1:a -c:v copy -shortest \"{}\"".format(tempfolderpath.format("_2.mp4"), pathtoaudio, outputfilepath.format("VIDEO", ".mp4")))

    #remove temporary files
    try:
        shutil.rmtree(os.path.join(outputfolder, "TEMP\\"))
    except Exception as e:
        print("\nCould not remove temp directory")
    
    #done
    print("\n{0} took {1:.2f}s total".format(os.path.basename(pathtoaudio), (time.time() - int(outfilename))))


if __name__ == "__main__":
    
    #print title
    print("============================ Music Visualiser V3.0 ============================")
    print("========================== Copyright PointZero 2022 ===========================")
    print("======== Do not distribute app. Output licensed under CC BY-NC-ND 4.0. ========")
    print("================== This build licensed to: ShibbySays community ================")
    print("!!!!!! THIS APP AND ITS OUTPUT IS EXCEEDINGLY LIKELY TO TRIGGER EPILEPSY !!!!!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!! PROCEED AT YOUR OWN RISK! !!!!!!!!!!!!!!!!!!!!!!!!!!")
    print()


    #imports
    from PIL import Image
    import cv2
    import scipy
    from scipy import signal
    from scipy.io.wavfile import read
    import numpy as np
    import time
    import psutil
    import os
    from shutil import move
    import argparse
    import data
    #from pyshader import python2shader, ivec3, i16, u8, Array
    import sys
    import shutil
    import subprocess

    if DEBUG:
        os.environ["RUST_BACKTRACE"] = "full"
        np.seterr(divide="ignore", invalid="ignore")
    else:
        np.seterr(divide="ignore", invalid="ignore")

    main()