# -*- coding: latin-1 -*-
# import the necessary packages
from __future__ import print_function
from PIL import Image
from PIL import ImageTk
import Tkinter as tki
import tkFileDialog, tkMessageBox
import threading
import datetime
import imutils
import cv2
import os
from imutils.video import VideoStream
import pyrealsense2 as rs
import sensor_utils
import sys
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
print(BASE_DIR, ROOT_DIR)
sys.path.append(ROOT_DIR) # config
sys.path.append(os.path.join(ROOT_DIR, 'utils')) # utils
sys.path.append(os.path.join(ROOT_DIR, 'libs')) # libs
from model_pose_ren import ModelPoseREN
from util import get_center_fast as get_center
from graphic_utils import drawAngles, nameangles, segmentImage, show_results, skeletonMeasures
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import xarray as xr
import subprocess
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import shutil

class PhotoBoothApp:
    def __init__(self, outputPath):
        # store the video stream object and output path, then initialize
        # the most recently read frame, thread for reading frames, and
        # the thread stop event
        self.bag = None
        self.params = dict()
        self.outputPath = outputPath
        self.frame = None
        self.thread = None
        self.stopEvent = None
        self.lower_ = 1
        self.pipeline = rs.pipeline()
        self.minmeasurementsWid = dict()
        self.maxmeasurementsWid = dict()
        self.crtmeasurementsWid = dict()
        self.minmeasurements = dict()
        self.maxmeasurements = dict()
        self.measurements = dict()
        self.continuePlotting = False
        self.isControl = 0
        self.params["patientnr"] = 0
        self.params["controlnr"] = 0
        self.sequencenr = 0
        self.counter = 1
        self.movtype = "f"
        self.filename = ""
        config = rs.config()
        #if rs.device_list().size() == 0:
        #    bag = "/media/cejnog/DATA/data/02-2019/bags/2019-02-02_12-05-14.bag"
        #    rs.config.enable_device_from_file(config, bag)
        # Start streaming
        #intr = intrinsics(pipeline, config)
        profile = self.pipeline.start(config)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()    # init hand pose estimation model
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        self.depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics        
        # intrinsic paramters of Intel Realsense SR300]
        self.dataset = "hands17"
        fx, fy, ux, uy = 463.889, 463.889, 320, 240        
        self.hand_model = ModelPoseREN(self.dataset, lambda img: get_center(img, lower=self.lower_, 
            upper=self.upper_.get()), param=(fx, fy, ux, uy), use_gpu=True)

        # initialize the root window and image panel
        self.root = tki.Tk()
        self.panel = None
        self.recording = False
        self.playback = False
        # create a button, that when pressed, will take the current
        # frame and save it to file
        self.frame = tki.Frame(self.root)
        self.frame.grid(row=0, column=0)
        labelframe = tki.LabelFrame(self.frame, text="Opções de gravação")
        labelframe.pack(fill="both", expand="yes")
 
        self.tableFrame = tki.LabelFrame(self.root, text="Medidas")
        
        self.acquisitionPath = "data/09-2019/bags/"
        with open(self.acquisitionPath + "config") as config:
            for lines in config:
                l = lines.split("=")
                self.params[l[0]] = int(l[1])
        self.btn = tki.Button(labelframe, text="Iniciar playback", 
            command=self.startPlayback)
        self.btn.pack(side="left", expand="no", padx=10,
            pady=10)
        self.btn2 = tki.Button(labelframe, text="Iniciar gravação",
            command=self.takeSnapshot)
        self.btn2.pack(side="left", expand="no", padx=10,
            pady=10)

        self.btn6 = tki.Button(labelframe, text="Novo paciente",
            command=self.metadata)
        self.btn6.pack(side="left", expand="no", padx=10,
            pady=10)
        self.lbl1 = tki.Label(labelframe, textvariable=self.params["patientnr"])
        self.lbl1.pack(side="left", expand="no", padx=10,
            pady=10)
        self.btn7 = tki.Button(labelframe, text="Nova sequência",
            command=self.newSeq)
        self.btn7.pack(side="left", expand="no", padx=10,
            pady=10)

        self.lbl2 = tki.Label(labelframe, textvariable=self.sequencenr)
        self.lbl2.pack(side="left", expand="no", padx=10,
            pady=10)
        
        self.textbtn = tki.Label(labelframe)
        self.textbtn.pack()

        labelHand = tki.LabelFrame(self.frame, text="Opções de tracking")
        labelHand.pack(fill="both", expand="yes")

        self.showAngles = tki.BooleanVar()
        self.showTracking = tki.BooleanVar()
        self.hand = tki.StringVar()
        self.upper_ = tki.IntVar()
        C1 = tki.Checkbutton(labelHand, text = "Mostrar ângulos", variable = self.showAngles, \
                 onvalue = True, offvalue = False)
        C1.pack(side = "left", expand = "no")
        R1 = tki.Radiobutton(labelHand, text="Mão Esquerda", variable=self.hand, value="l")
        R1.pack( anchor = tki.W, side = "left", expand = "no")

        R2 = tki.Radiobutton(labelHand, text="Mão Direita", variable=self.hand, value="r")
        R2.pack(side = "left", expand = "no")

        R3 = tki.Checkbutton(labelHand, text="Mostrar tracking", variable=self.showTracking, onvalue = True, offvalue = False)
        R3.pack(side = "left", expand = "no")
        
        R4 = tki.Scale(labelHand, from_=200, to=750,tickinterval=25, variable = self.upper_, orient=tki.HORIZONTAL, command=self.initPoseREN)
        R4.set(400)
        R4.pack(side = "left", expand = "no")
        

        self.lbl3 = tki.Label(labelframe, textvariable=self.filename)
        self.lbl3.pack(side="left", expand="no", padx=10,
            pady=10)

        height = len(nameangles)+1
        width = 3
        b = tki.Label(self.tableFrame, text="Ângulo", font=('Arial', 20))
        b.grid(row=0, column=0)
        b = tki.Label(self.tableFrame, text="Min", font=('Arial', 20))
        b.grid(row=0, column=1)
        b = tki.Label(self.tableFrame, text="Max", font=('Arial', 20))
        b.grid(row=0, column=2)
        b = tki.Label(self.tableFrame, text="Atual", font=('Arial', 20))
        b.grid(row=0, column=3)
        for i in range(1, height): #Rows
            b = tki.Label(self.tableFrame, text=nameangles[i-1], font=('Arial', 18))
            b.grid(row=i, column=0)
            self.minmeasurementsWid[nameangles[i-1]] = tki.Label(self.tableFrame, text="", font=('Arial', 18))
            self.minmeasurementsWid[nameangles[i-1]].grid(row=i, column=1)
            self.maxmeasurementsWid[nameangles[i-1]] = tki.Label(self.tableFrame, text="", font=('Arial', 18))
            self.maxmeasurementsWid[nameangles[i-1]].grid(row=i, column=2)
            self.crtmeasurementsWid[nameangles[i-1]] = tki.Label(self.tableFrame, text="", font=('Arial', 20))
            self.crtmeasurementsWid[nameangles[i-1]].grid(row=i, column=3)
            
            self.minmeasurements[nameangles[i-1]] = 9999
            self.maxmeasurements[nameangles[i-1]] = -9999
            self.measurements[nameangles[i-1]] = list()

        self.btn3 = tki.Button(self.tableFrame, text="Plotar gráficos", 
            command=self.showPlots)
        self.btn3.grid(row=height+1, column=1)

        self.btn4 = tki.Button(self.tableFrame, text="Gerar resultados", 
            command=self.execScript)
        self.btn4.grid(row=height+1, column=2)

        self.btn5 = tki.Button(self.tableFrame, text="Resetar tabela de medidas", 
            command=self.reset)
        self.btn5.grid(row=height+1, column=3)

        self.label = tki.Label(labelHand)
        self.label.pack()
        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()
        self.orth = tki.IntVar()
        

        # set a callback to handle when the window is closed
        self.root.wm_title("Hand Record/Playback Tool")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

    def videoLoop(self):
        # DISCLAIMER:
        # I'm not a GUI developer, nor do I even pretend to be. This
        # try/except statement is a pretty ugly hack to get around
        # a RunTime error that Tkinter throws due to threading
        try:
            # keep looping over frames until we are instructed to stop
            while not self.stopEvent.is_set():
                # grab the frame from the video stream and resize it to
                # have a maximum width of 300 pixels
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                #if not depth_frame:
                #    return None
                # Convert images to numpy arrays
                depth_image = np.asarray(depth_frame.get_data(), dtype=np.float32)    
                depth = depth_image * self.depth_scale * 1000
                depth[depth == 0] = depth.max()
                #img = np.minimum(depth, 1500)
                #img = (img - img.min()) / (img.max() - img.min())
                #img = np.uint8(img*255)
                #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)                                    
                
                if self.showTracking.get():
                    if self.hand.get() == 'l':
                        depth = depth[:, ::-1]  # flip
                    results, _ = self.hand_model.detect_image(depth)
                    if self.showAngles.get():
                        self.counter = 0
                        self.tableFrame.grid(row=0, column=1)
                        img_show = drawAngles(depth, results, self.dataset, -1, self.depth_intrin)
                        angles, positions = skeletonMeasures(results, 'hands17', self.depth_intrin)
                        for angle in nameangles:
                            if angles[angle] < self.minmeasurements[angle]:
                                self.minmeasurements[angle] = angles[angle]
                                self.minmeasurementsWid[angle].config(text=str(int(self.minmeasurements[angle])))
                            if angles[angle] > self.maxmeasurements[angle]:
                                self.maxmeasurements[angle] = angles[angle]
                                self.maxmeasurementsWid[angle].config(text=str(int(self.maxmeasurements[angle])))
                            self.crtmeasurementsWid[angle].config(text=str(int(angles[angle])))
                            self.measurements[angle].append(angles[angle])
                    else:
                        if self.counter == 0:
                            self.tableFrame.grid_remove()    
                            for angle in nameangles:
                                self.minmeasurements[angle] = 9999
                                self.maxmeasurements[angle] = -9999
                                self.measurements[angle] = list()
                        self.counter = self.counter + 1
                    img = np.minimum(depth, self.upper_.get())
                    img = (img - img.min()) / (img.max() - img.min())
                    
                    img_show = show_results(img, results, self.dataset)
                    if self.hand.get() == 'l':
                        img_show = img_show[:, ::-1, :]  # flip
                    
                else:
                    img = np.minimum(depth, self.upper_.get())
                    img = (img - img.min()) / (img.max() - img.min())
                    img = np.uint8(img*255)
                    img_show = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    
        

                img_show = cv2.resize(img_show, (800, 600))
                # OpenCV represents images in BGR order; however PIL
                # represents images in RGB order, so we need to swap
                # the channels, then convert to PIL and ImageTk format
                #image = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(img_show)
                image = ImageTk.PhotoImage(image)
        
                # if the panel is not None, we need to initialize it
                if self.panel is None:
                    self.panel = tki.Label(self.frame, image=image)
                    self.panel.image = image
                    self.panel.pack(side="left", padx=10, pady=10)
        
                # otherwise, simply update the panel
                else:
                    self.panel.configure(image=image)
                    self.panel.image = image

        except RuntimeError:
            print("[INFO] caught a RuntimeError")

    def execScript(self):
        if self.bag == None:
            self.bag = tkFileDialog.askopenfilename(initialdir = "/home/cejnog/Documents/Pose-REN/data/",title = "Select file",filetypes = (("bag files","*.bag"),("all files","*.*")))
        subprocess.call("./processBag.sh %s %s" % (self.bag, self.hand.get()), shell=True)
        return

    def reset(self):
        for angle in nameangles:
            self.minmeasurements[angle] = 9999
            self.maxmeasurements[angle] = -9999
            self.measurements[angle] = list()
        return
    
    def metadata(self):
        if self.isControl:
            self.params["controlnr"] += 1
        else:
            self.params["patientnr"] += 1
        self.sequencenr = 0
        self.newSeq()

    def newSeq(self):
        l = "P" + str(self.params["patientnr"]) + "_" + self.movtype
        self.filename = l + self.hand.get() + str(self.sequencenr) + ".bag"
        self.sequencenr += 1
        self.lbl3.config(text=self.filename)
        #self.auxw = tki.Toplevel()
        #self.metadata = tki.LabelFrame(self.auxw, text="Metadados")
        #self.metadata.pack(fill="both", expand="yes")
        #self.lname = tki.Label(self.metadata, text="Nome:").grid(row=0, column=0)
        #self.pname = tki.Entry(self.metadata)
        #self.pname.grid(row=0, column=1)
        #self.lfeed = tki.Label(self.metadata, text="Observações:").grid(row=2, column=0)
        #self.feedb = tki.Text(self.metadata, height=20, width=30)
        #self.feedb.grid(row=2, column=1, padx=10, pady=10)
        #self.orthBut = tki.Checkbutton(self.metadata, text="Paciente usa órtese?", variable=self.orth).grid(row=1, column = 1)
        #self.btn7 = tki.Button(self.metadata, text="Confirmar", command=self.setpatientdata).grid(row=3, column=1)

    def setpatientdata(self):
        self.obs = self.feedb.get("1.0","end-1c")
        self.name = self.pname.get()
        self.orthosis = self.orth.get()
        ts = datetime.datetime.now()
        self.date = ts
        filename = "data/{}.bag".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
        p = os.path.sep.join((self.outputPath, filename))
        self.filename = p
        self.auxw.destroy()
        print(self.name, self.obs, self.date, self.filename)

    def startPlayback(self):
        if not(self.playback):
            self.bag = tkFileDialog.askopenfilename(initialdir = "/home/cejnog/Documents/Pose-REN/data/",title = "Select file",filetypes = (("bag files","*.bag"),("xml files","*.xml"),("all files","*.*")))
            self.pipeline.stop()
            config = rs.config()
            rs.config.enable_device_from_file(config, self.bag)
            profile = self.pipeline.start(config)        
            self.btn.config(text = "Parar Playback (voltar para a camera)")
        else:
            self.pipeline.stop()
            # Create a config object
            config = rs.config()
            
            profile = self.pipeline.start(config)                
            self.btn.config(text = "Iniciar Playback")
        self.playback = not(self.playback)  

    def takeSnapshot(self):        
        if not(self.recording):
            # grab the current timestamp and use it to construct the
            # output path            
            #ts = datetime.datetime.now()
            #filename = "{}.bag".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
            #p = os.path.sep.join((self.outputPath, filename))
            if not(self.filename):
                self.metadata()
            self.pipeline.stop()
            config = rs.config()
            config.enable_record_to_file(self.filename)
            profile = self.pipeline.start(config)        
            self.btn2.config(text = "Parar Gravação")
        else:
            self.pipeline.stop()
            #self.createxml()
            config = rs.config()
            
            threading.Thread(target=shutil.move(self.filename, self.acquisitionPath)).start()
            self.newSeq()
            profile = self.pipeline.start(config)    
            self.btn2.config(text = "Iniciar Gravação")    
        self.recording = not(self.recording)            
    
    def createxml(self):
        #root = ET.Element("root")
        patient = ET.Element("patient")
        ET.SubElement(patient, "name").text = self.name
        ET.SubElement(patient, "feedback").text = self.obs
        ET.SubElement(patient, "orthosis").text = self.orth.get()
        ET.SubElement(patient, "filename").text = self.filename
        tree = ET.ElementTree(patient)
        tree.write(self.filename.replace(".bag", ".xml"))      

    def showPlots(self):
        self.continuePlotting = not(self.continuePlotting)
        
        self.graphlabel = tki.LabelFrame(self.root, text="Gráficos")
        self.graphlabel.grid(row=1, column=0, columnspan=2)
        fig2 = plt.figure(figsize=(16,4))
        cols = 10
        if len(nameangles) % cols == 0:
            sum = 0
        else:
            sum = 1
        
        f = fig2.subplots(sum + (len(nameangles)/cols), cols)
        for i in range(len(nameangles)):
            f[i/cols, i%cols].set_title(nameangles[i])
            f[i/cols, i%cols].cla()
            f[i/cols, i%cols].grid()
                    
        fig2.tight_layout()
        graph = FigureCanvasTkAgg(fig2, master=self.graphlabel)
        graph.get_tk_widget().pack(side="top",fill='both',expand=True)
            
        def plotter():
            dpts = list()    
            while self.continuePlotting:
                for i in range(len(nameangles)):
                    # print(len(angle[:][i]))
                    # f = fig2.add_subplot(spec2[i/cols, i%cols])
                    f[i/cols, i%cols].cla()
                    f[i/cols, i%cols].grid()
                    f[i/cols, i%cols].set_title(nameangles[i])
                    f[i/cols, i%cols].plot(self.measurements[nameangles[i]], '-', label=str(i)) 
                graph.draw()
                fig2.tight_layout()
                time.sleep(0.05)
        threading.Thread(target=plotter).start()
        
    
    def initPoseREN(self, event=None):
        return
        #print(self.upper_.get())
        #fx, fy, ux, uy = 463.889, 463.889, 320, 240        
        #self.hand_model = ModelPoseREN(self.dataset, lambda img: get_center(img, lower=self.lower_, upper=self.upper_.get()), param=(fx, fy, ux, uy), use_gpu=True)

    def saveconfig(self):
        with open(self.acquisitionPath + "config", 'w') as configfile:
            for par in self.params:
                configfile.write(par + "=" + str(self.params[par])+"\n")

    def onClose(self):
        self.stopEvent.set()
        self.pipeline.stop()
        with open(self.acquisitionPath + "config", 'w') as configfile:
            for par in self.params:
                configfile.write(par + "=" + str(self.params[par])+"\n")

        self.root.destroy()

import argparse
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
    help="path to output directory to store snapshots")
args = vars(ap.parse_args())

# initialize the video stream and allow the camera sensor to warmup
time.sleep(2.0)

# start the app
pba = PhotoBoothApp(args["output"])
pba.root.mainloop()
