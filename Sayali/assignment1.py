import numpy as np
import cv2
from tkinter import filedialog
from tkinter import Tk, Button, Label, Frame, Menu, TOP, HORIZONTAL, LEFT, X, messagebox, Scale
from PIL import Image
from PIL import ImageTk


class ipGUI:
    def __init__(self, master):
        self.master = master
        self.master.minsize(width=1250, height=950)
        
        menu = Menu(self.master)
        master.config(menu=menu)

        # ***** Toolbar *****
        toolbar = Frame(master, bg="blue")
        undoButton = Button(toolbar, text="Undo", command=self.undoFunc)   # Button for undo function
        undoButton.pack(side=LEFT)
        openButton = Button(toolbar, text="Open", command=self.openImage)  # Button for opening image
        openButton.pack(side=LEFT)
        saveButton = Button(toolbar, text="Save", command=self.saveImage)  # Button for saving image
        saveButton.pack(side=LEFT)
        exitButton = Button(toolbar, text="Exit", command=master.destroy)  # Button for exiting out from gui window
        exitButton.pack(side=LEFT)
        hsButton = Button(toolbar, text="Histogram Equalization", command=self.histogram) # Button for histogram equilization
        hsButton.pack(side=LEFT)
        gsButton = Button(toolbar, text="Gamma Correction for pow<1", command=self.gamma) # Button for Gamma Correction with pow < 1
        gsButton.pack(side=LEFT)
        xsButton = Button(toolbar , text = "Gamma Correction for pow>1", command = self.gamma1) #Button for gamma correction with pow >1 
        xsButton.pack(side=LEFT)
        lsButton = Button(toolbar , text = "Log Transform", command = self.logTrans) # Button for log transformation
        lsButton.pack(side=LEFT)
        psButton = Button(toolbar, text="Blur", command=self.blur) # Button for blur function
        psButton.pack(side=LEFT)
        tsButton = Button(toolbar, text="Sharp", command=self.sharp) # Button for sharp function
        tsButton.pack(side=LEFT)
        jsButton = Button(toolbar, text="Edge Detection", command=self.Edge) # Button for edge detection
        jsButton.pack(side=LEFT)
        bsButton = Button(toolbar, text="Undo All", command=self.undoAll)   # Button for undo all
        bsButton.pack(side=LEFT)
        
        toolbar.pack(side=TOP, fill=X)
        
         # ***** Image Display Area *****
        self.frame = Frame(self.master)
        self.frame.pack()
        self.panel = Label(self.frame)
        self.panel.pack(padx=10, pady=10)
        self.img = None
        self.origImg = None
        self.prevImg = None


        #***** Box Blur Controls *****
        self.boxFrame = Frame(self.master)
        self.boxSlider = Scale(self.boxFrame, from_=1, to=5, orient=HORIZONTAL) # scale slider
        self.boxSlider.pack(side=TOP)
        self.boxExitButton = Button(self.boxFrame, text="exit", command=self.boxFrame.pack_forget) # button for exiting scale slider
        self.boxExitButton.pack(side=TOP)
        

        # ***** DFT Display Area ******
        self.dftFrame = Frame(self.master)
        self.magPanel = Label(self.dftFrame)
        self.magPanel.pack(padx=10, pady=10,side=TOP)
        self.freqPanel = Label(self.dftFrame)
        self.freqPanel.pack(padx=10, pady=10,side=TOP)
        self.dftExitButton = Button(self.dftFrame, text="exit", command=lambda: self.displayImg(np.array(self.img)))
        self.dftExitButton.pack(side=TOP, fill=X)

        # ***** Low Pass Mask Creation *****
        self.lpmFrame = Frame(self.master)
        self.lpmPanel = Label(self.lpmFrame)
        self.lpmPanel.pack(padx=10, pady=10,side=TOP)
        self.lpmSubButton = Button(self.lpmFrame, text="submit")
        self.lpmSubButton.pack(side=TOP)
        self.lpmExitButton = Button(self.lpmFrame, text="exit", command=lambda: self.displayImg(np.array(self.img)))
        self.lpmExitButton.pack(side=TOP)
        self.lpmSlider = Scale(self.lpmFrame, from_=1, to=2, orient=HORIZONTAL, resolution=1)

        # ***** High Pass Mask Creation *****
        self.hpmFrame = Frame(self.master)
        self.hpmPanel = Label(self.hpmFrame)
        self.hpmPanel.pack(padx=10, pady=10,side=TOP)
        self.hpmSubButton = Button(self.hpmFrame, text="submit")
        self.hpmSubButton.pack(side=TOP)
        self.hpmExitButton = Button(self.hpmFrame, text="exit", command=lambda: self.displayImg(np.array(self.img)))
        self.hpmExitButton.pack(side=TOP)
        self.hpmSlider = Scale(self.hpmFrame, from_=1, to=2, orient=HORIZONTAL, resolution=1)

        # ***** Band Pass Mask Creation *****
        self.bppmFrame = Frame(self.master)
        self.bppmPanel = Label(self.bppmFrame)
        self.bppmPanel.pack(padx=10, pady=10,side=TOP)
        self.bppmSubButton = Button(self.bppmFrame, text="submit")
        self.bppmSubButton.pack(side=TOP)
        self.bppmExitButton = Button(self.bppmFrame, text="exit", command=lambda: self.displayImg(np.array(self.img)))
        self.bppmExitButton.pack(side=TOP)
        self.bppmSliderLow = Scale(self.bppmFrame, from_=1, to=2, orient=HORIZONTAL, resolution=1)
        self.bppmSliderHigh = Scale(self.bppmFrame, from_=1, to=2, orient=HORIZONTAL, resolution=1)

        # ***** Band Pass Mask Creation *****
        self.bspmFrame = Frame(self.master)
        self.bspmPanel = Label(self.bspmFrame)
        self.bspmPanel.pack(padx=10, pady=10,side=TOP)
        self.bspmSubButton = Button(self.bspmFrame, text="submit")
        self.bspmSubButton.pack(side=TOP)
        self.bspmExitButton = Button(self.bspmFrame, text="exit", command=lambda: self.displayImg(np.array(self.img)))
        self.bspmExitButton.pack(side=TOP)
        self.bspmSliderLow = Scale(self.bspmFrame, from_=1, to=2, orient=HORIZONTAL, resolution=1)
        self.bspmSliderHigh = Scale(self.bspmFrame, from_=1, to=2, orient=HORIZONTAL, resolution=1)
        
    def displayImg(self, img):
        # input image in RGB
        self.frame.pack()
        self.dftFrame.pack_forget()
        self.lpmFrame.pack_forget()
        self.hpmFrame.pack_forget()
        self.bppmFrame.pack_forget()
        self.bspmFrame.pack_forget()
        self.img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(self.img)

        self.panel.configure(image=imgtk)
        self.panel.image = imgtk

    def openImage(self):
        # can change the image
        path = filedialog.askopenfilename()       # to get image from computer location
        if len(path) > 0:                         # condition given for image is selected or not
            imgRead = cv2.imread(path)            # for reading file path
            if imgRead is not None:
                imgRead = cv2.cvtColor(imgRead, cv2.COLOR_BGR2RGB)
                self.origImg = Image.fromarray(imgRead)   # storing original image
                self.prevImg = Image.fromarray(imgRead)   # storing previous image
                self.displayImg(imgRead)                  # displaying image
            else:
                raise ValueError("Not a valid image")     
        else:
            raise ValueError("Not a valid path")

    def saveImage(self):
        if self.img is not None:
            toSave = filedialog.asksaveasfilename()   #function from filedialog module to save image
            self.img.save(toSave)                    
        else:
            messagebox.showerror(title="Save Error", message="No image to be saved!")  # if image is not saved 
    
    
    
    def convolution(self,img, kernel):                                        # UDF(User Defined Fn) for convolution
          m, n = kernel.shape                                                 # Taking size of image and kernel
          y, x = img.shape[:2]
          new_image = np.zeros((y, x))                                        # Initializing new image
          y += 1 - m                                                          # reducing image to kernel size
          x += 1 - m
          for i in range(y):                                                  # Iterations for computing Convolution
             for j in range(x):
                new_image[i][j] = np.sum(img[i:i + m, j:j + m] * kernel)
          return new_image 


    def boxBlurFunc(self,event, img, filterSize):                             # UDF for blur call function 
          img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)                          # changing from RGB to HSV
          filt = (np.ones((filterSize, filterSize))/(filterSize**2))          # filter for blurring
          imgNew = img.copy()
          imgNew[:,:,2] = self.convolution(imgNew[:,:,2], filt)               # convolving to get output
          imgNew = cv2.cvtColor(imgNew, cv2.COLOR_HSV2RGB)                    # changing from HSV to RGB
          return imgNew
      
    def blur(self):                                                           # Execution function for for blur
        img = np.array(self.img)                                              
        self.prevImg = self.img
        self.boxFrame.pack()                                                  
        self.boxSlider.set(1)                                                 # slidder is set from 1
        self.boxSlider.bind("<ButtonRelease-1>", lambda event, img=img: self.displayImg # blurrng with the help of slidder
                            (self.boxBlurFunc(event, img, 2*self.boxSlider.get()+1)))

    
    def findHistogram(self,img):                                              
       # Histogram Equalization
       # img is assumed in hsv
       m,n = img.shape[:2]
       hist = np.zeros(256)
       for i in range(m):
          for j in range(n):
             hist[img[i][j][2]] += 1
       return hist

    def histEqlFunc(self,img):                                                 #  UDF for histogram equilization
        # img in RGB, return imgNew RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        m,n = img.shape[:2]
        hist =self.findHistogram(img)
        l = 256
        transform = (l-1)*np.array([sum(hist[:i+1]) for i in range(l)])/(m*n)  # command for equal weights
        transform = np.array(np.round(transform), dtype=np.uint8)
        imgNew = np.array([[[img[i][j][0], img[i][j][1], transform[img[i][j][2]]] for j in range(n)] for i in range(m)]) # making image pixels intensity to same level
        imgNew = cv2.cvtColor(imgNew, cv2.COLOR_HSV2RGB)
        return imgNew
    
    
    def histogram(self):
        img = np.array(self.img)
        self.prevImg = self.img
        imgNew = self.histEqlFunc(img)
        self.displayImg(imgNew)

    def gammaCorFunc(self,img, c=1.0, gamma=1.0):                        # UDF for gamma transform     
         img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
         m,n = img.shape[:2]
         transform = [min(c*(r**gamma), 255) for r in range(256)]        
         transform = np.array(np.round(transform), dtype=np.uint8)
         imgNew = np.array([[[img[i][j][0], img[i][j][1], transform[img[i][j][2]]] for j in range(n)] for i in range(m)])
         imgNew = cv2.cvtColor(imgNew, cv2.COLOR_HSV2RGB)
         return imgNew

    # Gamma Correction
    def gamma(self):
        img = np.array(self.img)
        self.prevImg = self.img
        imgNew = self.gammaCorFunc(img, 255.0/(255.0**0.5), 0.5)        #power transform for power = 0.5 
        self.displayImg(imgNew)
        
    def gamma1(self):
        img = np.array(self.img)
        self.prevImg = self.img
        imgNew = self.gammaCorFunc(img, 255.0/(255.0**2), 2)            #power transform for power = 2
        self.displayImg(imgNew)
        
    def logTranFunc(self,img, c=1.0):                                      # UDF for log transformation
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)                         # converting from RGB to HSV
        m,n = img.shape[:2]                                            
        transform = [min(c*np.log10(1+r),255) for r in range(256)]         # taking min value
        transform = np.array(np.round(transform), dtype=np.uint8)          # takkiing int of it
        imgNew = np.array([[[img[i][j][0], img[i][j][1], transform[img[i][j][2]]] for j in range(n)] for i in range(m)])
        imgNew = cv2.cvtColor(imgNew, cv2.COLOR_HSV2RGB)
        return imgNew

    def logTrans(self):                                                  # Execution function for log transform
        img = np.array(self.img)
        self.prevImg = self.img
        imgNew = self.logTranFunc(img, 255.0/np.log10(256))              # Calling log transformation function
        self.displayImg(imgNew)
    
    def sharpFunc(self,img):                                             # UDF for sharpening image
          img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)                      # converting from RGB to HSV
          filt = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])           # declaring filter
          filt = np.multiply(filt,2)
          imgNew = img.copy()
          imgNew[:, :, 2] = self.convolution(imgNew[:, :, 2], filt)       # convolving img and filter
          imgNew = cv2.cvtColor(imgNew, cv2.COLOR_HSV2RGB)                  # converting back to RGB
          return imgNew
         
    
    def sharp(self):                                                     # Execution function for sharpening
        img = np.array(self.img)
        self.prevImg = self.img
        imgNew = np.add(self.sharpFunc(img),img)                                      # Calling default sharp function
        imgNew = np.clip(imgNew, 0, 255)
        self.displayImg(imgNew)                                           # Displaying sharpened image
    
    def edgeFunc(self,img):                                              # UDF for edge detection 
         img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)                      # converting from RGB to HSV
         filt = np.array([[-1,-2,-1], [0, 0, 0], [1, 2, 1]])             # declaring filter for edge detection
         imgNew = img.copy()
         imgNew[:, :, 2] = self.convolution(imgNew[:, :, 2], filt)       # convolving with filter
         imgNew = cv2.cvtColor(imgNew, cv2.COLOR_HSV2RGB)                # converting back to RGB
         return imgNew
     
    def Edge(self):                                                   # Execution function for edge detection
        img = np.array(self.img)
        self.prevImg = self.img
        imgNew = self.edgeFunc(img)                                    # calling edge function
        self.displayImg(imgNew)                                        # Displaying edge detected image

    def undoFunc(self):                                               # function for undoing to previous image
        self.img = self.prevImg
        self.displayImg(np.array(self.img))                            # displaying previous image when function is called
    
    def undoAll(self):                                               # function for undoing all 
        self.img = self.origImg                                       # original image is stored in it
        self.displayImg(np.array(self.img))                          # displaying original image when function is called
root = Tk()
ipGUI(root)
root.mainloop()
