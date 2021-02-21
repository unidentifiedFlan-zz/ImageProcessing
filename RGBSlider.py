from PIL import Image, ImageTk
import tkinter as tk
import numpy as np

window = tk.Tk()

class colour_balancer:

    def __init__(self):

        self.im = Image.open('C:/Users/scday/Pictures/Random/shaunofthedead.png')
        self.im.format = 'RGB'

        self.r_fac = tk.Scale(window, from_=0, to=5, orient=tk.HORIZONTAL, command=self.updateImg)
        self.r_fac.set(1)
        self.g_fac = tk.Scale(window, from_=0, to=5, orient=tk.HORIZONTAL, command=self.updateImg)
        self.g_fac.set(1)
        self.b_fac = tk.Scale(window, from_=0, to=5, orient=tk.HORIZONTAL, command=self.updateImg)
        self.b_fac.set(1)

        self.r_fac.pack()
        self.g_fac.pack()
        self.b_fac.pack()

        self.r_prev = self.r_fac.get()
        self.g_prev = self.g_fac.get()
        self.b_prev = self.b_fac.get()

        self.img = ImageTk.PhotoImage(self.im)
        self.lbl = tk.Label(window, image = self.img)
        self.lbl.pack()

    def updateImg(self, event):
        if (self.r_fac.get() != self.r_prev) or (self.g_fac.get() != self.g_prev) or (self.b_fac.get() != self.b_prev):
            self.r_prev = self.r_fac.get()
            self.g_prev = self.g_fac.get()
            self.b_prev = self.b_fac.get()

            arr = np.array(self.im)
            arr[:, :, 0] *= self.r_prev
            arr[:, :, 1] *= self.b_prev
            arr[:, :, 2] *= self.g_prev
            mergedImg = Image.fromarray(arr)

#            r, g, b = self.im.split()
#            r = r.point(lambda i: i*self.r_prev)
#            g = g.point(lambda i: i * self.g_prev)
#            b = b.point(lambda i: i * self.b_prev)
#            mergedImg = Image.merge('RGB', (r, g, b))

            self.img = ImageTk.PhotoImage(mergedImg)
            self.lbl.configure(image=self.img)
            self.lbl.image = self.img



balancer = colour_balancer()
window.mainloop()