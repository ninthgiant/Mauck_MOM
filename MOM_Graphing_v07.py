##############################
#
#    MOM_Graphing_07.py - RAM, June 8, 2024
#       contains graphing functions for main app
#     
#######################################

#######################################
# Libraries setup
#######################################
import matplotlib.pyplot as plt
import matplotlib.backend_bases as backendgi
import numpy as np

#############################
#   DraggableMarker:  A class for a set of draggable markers on a Matplotlib-plt line plot
#   Designed to record data from two separate markers, which the user confirms with an "enter key"
#   Adapted by Liam Taylor from https://stackoverflow.com/questions/43982250/draggable-markers-in-matplotlib
#############
class DraggableMarker():
    def __init__(self, category, startY, startX=0):
        self.isGood = False
        self.category = category

        self.index_start = 0
        self.index_end = 0

        self.buttonClassIndex = 0
        self.buttonClasses = ["{category} start".format(category=category), "{category} end".format(category=category)]

        self.ax = plt.gca()  # this assumes a current figure object? Is this the only external assumption?
        self.lines=self.ax.lines
        self.lines=self.lines[:]

        self.tx = [self.ax.text(0,0,"") for l in self.lines]
        self.marker = [self.ax.plot([startX],[startY], marker="o", color="red")[0]]

        self.draggable = False

        self.isZooming = False
        self.isPanning = False

        self.currX = 0
        self.currY = 0

        self.c0 = self.ax.figure.canvas.mpl_connect("key_press_event", self.key)
        self.c1 = self.ax.figure.canvas.mpl_connect("button_press_event", self.click)
        self.c2 = self.ax.figure.canvas.mpl_connect("button_release_event", self.release)
        self.c3 = self.ax.figure.canvas.mpl_connect("motion_notify_event", self.drag)

    def click(self,event):
        if event.button==1 and not self.isPanning and not self.isZooming:
            #leftclick
            self.draggable=True
            self.update(event)
            [tx.set_visible(self.draggable) for tx in self.tx]
            [m.set_visible(self.draggable) for m in self.marker]
            self.ax.figure.canvas.draw_idle()        
                
    def drag(self, event):
        if self.draggable:
            self.update(event)
            self.ax.figure.canvas.draw_idle()

    def release(self,event):
        self.draggable=False
        
    def update(self, event):
        try:        
            line = self.lines[0]
            x,y = self.get_closest(line, event.xdata) 
            self.tx[0].set_position((x,y))
            self.tx[0].set_text(self.buttonClasses[self.buttonClassIndex])
            self.marker[0].set_data([x],[y])
            self.currX = x
            self.currY = y
        except TypeError:
            pass

    def get_closest(self,line, mx):
        x,y = line.get_data()
        try: 
            mini = np.argmin(np.abs(x-mx))
            return x[mini], y[mini]
        except TypeError:
            pass

    ##############
    # Not sure we even want these - would rather deal with buttons on the window
    #####
    def key(self,event):
        if (event.key == 'o'):
            self.isZooming = not self.isZooming
            self.isPanning = False
        elif(event.key == 'p'):
            self.isPanning = not self.isPanning
            self.isZooming = False
        elif(event.key == 't'):
            # A custom re-zoom, now that 'r' goes to 
            # the opening view (which might be retained from a previous view)
            line = self.lines[0]
            full_xstart = min(line.get_xdata())
            full_xend = max(line.get_xdata())
            full_ystart = min(line.get_ydata())
            full_yend = max(line.get_ydata())
            self.ax.axis(xmin=full_xstart, xmax=full_xend, ymin=full_ystart, ymax=full_yend)
        elif (event.key == 'enter'):  #### these are the event keys we need for now
            if(self.buttonClassIndex==0):
                self.ax.plot([self.currX],[self.currY], marker="o", color="yellow")
                self.buttonClassIndex=1
                self.index_start = self.currX
                plt.title("Add {category} end point, then press enter.".format(category=self.category))
            elif(self.buttonClassIndex==1):
                self.index_end = self.currX
                self.isGood = True
                plt.close()
            self.update(event)

###################
#   AxesLimits: A class defining an object that stores axes limits for
#       pyplot displays
#######
class AxesLimits():
    def __init__(self, xstart, xend, ystart, yend):
        self.xstart = xstart
        self.xend = xend
        self.ystart = ystart
        self.yend = yend



#########################
#   annotateCurrentMarkers: A function to plot all markers from a markers dataframe on the current plt viewer
#       Original code from Liam Taylor converted to a method
#       (to be used for the markers dataframe as returned by getTracePointPair)
########
def annotateCurrentMarkers(markers):
    ax = plt.gca() # this assumes a current figure object? Is this the only external assumption?

    # Plot the pairs of marker points separately, so lines aren't drawn betwen them
    for l, df in markers.groupby("Category"):
        ax.plot(df.loc[:,"Measure"], marker="o", color="black", ms=8)
        for index, row in df.iterrows():
            label = "{category} {point}".format(category=df.loc[index,"Category"], point=df.loc[index, "Point"])
            ax.annotate(label, (index, df.loc[index, "Measure"]), rotation=60)
