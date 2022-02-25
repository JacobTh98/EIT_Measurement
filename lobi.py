import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import os
from datetime import datetime
import serial
from tqdm import tqdm
from PIL import Image
import pyglet
import reconstruction #not installable with pip. It´s a package from OpenEIT
import imageio

font = {'family': 'DejaVu Sans',
        'color':  'black',
        'weight': 'normal',
        'size': 24,}

"""
Um dieses Modul nach einer Veränderung zu verwenden, muss der Kernel neu gestartet werden.
"""
###--------------------------------------------------------
def gen_env(mes,el,mod,Schnkstp,leitw,temp, wasst, sonst='Keine weiteren Angaben'):
    """
    Initialisierung des Ordners mit Informationen zur Messung
    Es müssen alle Informationen übergeben werden.
    Input:  - Elektrodenanzahl [16/32]
            - Messmodus: ['a','b','c','d','e']
            - Schrittweite Schunk
            - Leitfähigkeit des Wassers
            - Temperatur der Umgebung
            - Wasserstand im Messzylinder
            - Sonstige Informationen zur Messung
    """
    print('Ordner mit dem Namen:"',mes,'" wurde erstellt.')
    os.makedirs(mes)
    dr = mes+'/info.txt'
    f = open(dr,"w")
    #Schreiben der Grunddaten
    f.write("EIT-Messung\n")
    f.write("----------------------------------\n")
    d = "Datum: \t \t \t"+str(datetime.today().strftime('%d.%m.%Y'))+"\n"
    u = "Uhrzeit: \t \t"+str(datetime.today().strftime('%H:%M'))+" Uhr\n"
    m = "Messmodus: \t \t" + str(mod) + "\n"
    e = "Elektrodenanzahl: \t" +str(el)+"\n"
    s = "Schunk Step Intevall \t" + str(Schnkstp)+"\t[°/step]"+"\n"
    l = "Leitfähigkeit \t \t" + str(leitw)+"\t[mS]\n"
    w = "Wasserstand \t \t" + str(wasst)+"\t[mm]\n"
    t = "Raumtemperatur \t \t"+ str(temp) + "\t[°C] \n \n"
    i = "Sonstige Informationen:\n -" + str(sonst)
    f.write(d);f.write(u);f.write(m);f.write(e)
    f.write(s);f.write(l);f.write(t);f.write(w)
    f.write(i)
    f.close()

###--------------------------------------------------------
def init(port="COM7"):
    """
    default port = COM7
    """
    serialPort = serial.Serial(port=port, baudrate=115200, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)
    print("Verbindung zu:" , port, "erfolgreich hergestellt.")
    return serialPort
###--------------------------------------------------------
def parse_line(line):
    """
    Aus pyEIT zu verarbeitung der bitdaten
    """
    try:
        _, data = line.split(":", 1)
    except ValueError:
        return None
    items = []
    for item in data.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            items.append(float(item))
        except ValueError:
            return None
    return np.array(items)
###--------------------------------------------------------
def measure_data(N,serialPort, M = 192, mode = 'e'):
    """
    Aufnahme von N Messwerten in einem definiterten Modus
    Input:  N    ... Anzahl der Messungen
            mode ... Modus des Spectra Kit
    Info:
            M+1 x N   ... Matrix, erste Spalte Messnummer.
            M         ... länge der Messung (default: 192)
            Matrix: 0 ... Messwerte
                    1 ... Messwerte
    Rückgabe:
            M+1 x N   ... Matrix, erste Spalte Messnummer.
    """
    n = np.arange(N) #Messnummerieung (0 ... N-1)
    MxN = np.zeros((N,M))
    #MxN = np.c_[n,MxN]
    cnt = 0
    while cnt < N:
        print("Vorgang: ",cnt+1,"von: ",N)
        line = serialPort.readline().decode("utf-8")
        try: #Probably a little bit buggy
            line = parse_line(line)
            l = len(line)
        except ValueError:
            l = 0
            print("Läenge unzureichend starte neu")
        if l == M:
            MxN[cnt,:]=line
            cnt = cnt+1
    #Data handling
    return MxN
###--------------------------------------------------------
def export_xlsx(A, path, mean, name ='undefined'):
    """
    Exportieren der messdaten als Excel-Tebelle
    !Achtung: Bei der Weiterverarbeitung!
    Außerdem wird ein mittelwertbild abgezogen
    - raw ... Rohdaten ohne Abzug des Mittelwertes
    - m_m ... Rohdaten inklusive Abzug des Mittelwertes
    """  
    ##Rohdaten
    df = pd.DataFrame(A)
    df.to_excel(excel_writer = str(path)+'/'+str(name)+'raw'+'.xlsx')
    np.save(str(path)+'/'+str(name)+'raw', A)
    ##Abzug des mittelwertes
    A = A - mean
    A[A<0]=0
    df = pd.DataFrame(A)
    df.to_excel(excel_writer = str(path)+'/'+str(name)+'m_m'+'.xlsx')
    np.save(str(path)+'/'+str(name)+'m_m', A)
    
    
    print('Messung',str(name),'erfolgreich exportiert')
###--------------------------------------------------------
def ground_truth(objct , r , α , path ,clockdirection=False, save_img = True):
    """
    Input: objct ...'rectangle','circle','triangle'
           r    ... Radius 0...2π [Rad] vom Mittelpunkt
           α    ... Winkel [°] von positiver x-Achse (default: gegen den Uhrzeigersinn)
           dr   ... Verzeichnis der Messng
           clockdirection ... im oder gegen Uhrzeigersinn drehen
           
           
           -> Für mehr Objekte Arrays für objct,r,α übergeben
    Return:
           Bild des Gegenstandes, positioniert im Einheitskreis.
           
           Change rotation cercle!!!
    """
    IMG=np.zeros((1000,1000))
    cv.circle(IMG,(500,500),500,(255,255,0),1) #Rand
    r_old = r
    α_old = α
    if type(r) == int:
        r = np.array([r])
        α = np.array([α])
        objct = np.array([objct])
        end = 1
    else:
        r = np.array(r)
        α = np.array(α)
        objct = np.array(objct)    
        end = len(r)
    
    for cnt in range(end):
        if clockdirection:
            α[cnt] = α[cnt]*(-1)# Grad in Radiant
        else:
            α[cnt] = α[cnt]     # Grad in Radiant
        if r[cnt]>100:
            print('r ist zu groß; Skalierbar von 0...100%')
            return
        r[cnt] = abs(r[cnt]*4)
        x_0 = y_0 = 500 #Verschiebung Nullpunkt Koordinatensystem
        x = round(r[cnt]*np.cos(α[cnt]))
        y = round(r[cnt]*np.sin(α[cnt]))
        if objct[cnt] == 'circle':
            cv.circle(IMG,(x_0+x,y_0+y),100,(255,0,0),-1)
        if objct[cnt] == 'rectangle':
            abw=100
            cv.rectangle(IMG,(x_0-abw+x,y_0-abw+y),(x_0+abw+x,y_0+abw+y),(255,0,0),-1)
        if objct[cnt] == 'triangle':
            pt1 = (500+x, 400+y)
            pt2 = (600+x, 600+y)
            pt3 = (400+x, 600+y)
            tri_edges = np.array( [pt1, pt2, pt3] )
            cv.drawContours(IMG, [tri_edges], 0, (255,0,0), -1)
    
    if save_img:
        im = Image.fromarray(IMG)
        im = im.convert("L")
        #im.save(path+"/"+str(objct)+str(r_old)+str(α_old)+".jpeg")
        im.save(path+"/"+"GroundTruth.jpeg")
        np.save(path+"/"+"GroundTruth_np",IMG)
        print('Bild gespeichert')
    return IMG
###--------------------------------------------------------

#
#
#
#
#
def view_txt(Dir):
    """
    Ausgeben der info.txt durch Übergabe des Speicherverzeichnisses
    """
    txt = Dir+'\info.txt'
    file = open(txt)
    print(file.read())
    
def view_GrTr(Dir):
    """
    Ausgabe der GroundTruth
    """
    Dir = Dir + '\GroundTruth_np.npy'
    img = np.load(Dir)
    plt.figure(figsize=(8,8))
    plt.title("GroundTruth", fontdict=font)
    plt.grid()
    plt.imshow(img)
    
def single_reconstruction(n_el,path,step,BackProj = True, diff_img = True,kind_of = 'm_m'):
    """
    n_el    ... Anzahl der Elektroden
    path    ... Verzeichnis der Messdaten
    step    ... Welche Iteration/Schritt wird ausgewertet
    BackProj... default: True, alternativ JacReconstruction
    diff_img... default: True = Es wird Ground Truth genommen, bei False zeros
    kind_of ... default: Meanfee: m_m. Alternative ist 'raw'
    """
    load_path = path+'/'+str(step)+kind_of+'.npy'
    IMGs = np.load(load_path)
    vis = IMGs[10,:]
    #vis = IMGs
    load_path = path +'/Mean_empty_ground.npy'
    GroundTruth = np.load(load_path)
    #Funktionier: 
    if BackProj:
        g = reconstruction.BpReconstruction(n_el=n_el)
    else:
        g =reconstruction.JacReconstruction(n_el=n_el)
    g.update_reference(GroundTruth)
    baseline = g.eit_reconstruction(vis)
    if diff_img:
        difference_image = g.eit_reconstruction(GroundTruth)
    else:
        difference_image = g.eit_reconstruction(np.zeros(len(GroundTruth)))#192
   
    mesh_obj = g.mesh_obj;el_pos = g.el_pos;ex_mat = g.ex_mat
    pts = g.mesh_obj['node'];tri = g.mesh_obj['element']
    x   = pts[:, 0];y   = pts[:, 1]
    #Print min and max impedance
    a = np.argmin(difference_image)
    b = np.argmax(difference_image)
    print('Minimaler Wert: ',difference_image[a])
    print('Maximaler Wert: ',difference_image[b])
    print('Δ Permittivität: ',np.abs(difference_image[a]-difference_image[b]))
    shading = 'gouraud'
    shading = 'flat'
    #SHOW # 
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.tripcolor(x , y , tri , difference_image, shading=shading, cmap=plt.cm.gnuplot)
    ax.plot(x[el_pos], y[el_pos], 'bo')
    for i, e in enumerate(el_pos):
        ax.text(x[e], y[e], str(i+1), size=12)
    ax.axis('equal')
    fig.colorbar(im)
    plt.show()
    
def gif_reconstruction(path, full=False , NameGif='Unnamed', BackProj = True, diff_img = True , kind_of = 'm_m'):
    """
    n_el     ... Anzahl der Elektroden __ wird selbst ausgelesen
    Y        ... Steps __ könnte selbst ausgelesen werden
    path     ... Verzeichnis der Messdaten
    steps    ... Welche Iteration/Schritt wird ausgewertet
    BackProj ... default: True, alternativ JacReconstruction
    diff_img ... default: True = Es wird Ground Truth genommen, bei False zeros
    kind_of  ... default: Meanfee: m_m. Alternative ist 'raw'
    """
    # Anzahl der Elektroden aus txt lesen:
    PATH = path+'/info.txt'
    with open(PATH, "r") as tf:
        lines = tf.read().split('\n')
    lin = lines[5]; 
    n_el = int(lin[int(len(lin))-2:])
    print("Elektrodenanzahl aus info.txt:\t",n_el)
    
    #Anzahl der Steps aus txt lesen:
    with open(PATH, "r") as tf:
        lines = tf.read().split('\n')
    lin = lines[6]; 
    lin = lin.split('\t')
    steps=int(lin[1])
    ### GIF reconstruction
    print("Step-Intervalle aus info.txt:\t",steps)
    #Y = np.arange(0,361,steps)
    Y = np.arange(0,126,steps)
    filenames = []
    toLoad = path+'/Mean_empty_ground.npy'
    GroundTruth = np.load(toLoad)
    if BackProj:
        g = reconstruction.BpReconstruction(n_el=n_el)
    else:
        g = reconstruction.JacReconstruction(n_el=n_el)
        
    if not full:
        for file_num in Y:
            ## Am Besten ist m_m zu Zero Reference
            toLoad = path+'/'+str(file_num)+'m_m.npy'
            IMGs = np.load(toLoad)
            vis = IMGs[IMGs.shape[0]//2,:] ###Take the rounded mid picture
            g.update_reference(GroundTruth)
            #g.update_reference(np.zeros(192))
            baseline = g.eit_reconstruction(vis)
            if diff_img:
                difference_image = g.eit_reconstruction(GroundTruth)
            else:
                difference_image = g.eit_reconstruction(np.zeros(192))

        #difference_image = g.eit_reconstruction(vis)
            mesh_obj = g.mesh_obj;el_pos = g.el_pos;ex_mat = g.ex_mat
            pts = g.mesh_obj['node'];tri = g.mesh_obj['element']
            x   = pts[:, 0];y = pts[:, 1]
            #Print min and max impedance
            a = np.argmin(difference_image)
            b = np.argmax(difference_image)
            #print('Minimaler Wert: ',difference_image[a])
            #print('Maximaler Wert: ',difference_image[b])
            #print('Δ Permittivität: ',np.abs(difference_image[a]-difference_image[b]))
            shading = 'gouraud'
            shading = 'flat'
            #SHOW # 
            #fig, ax = plt.subplots(figsize=(10, 8))
            plt.figure(figsize=(10,8))
            im = plt.tripcolor(x , y , tri , difference_image, shading=shading, cmap=plt.cm.gnuplot)
            plt.tripcolor(x , y , tri , difference_image, shading=shading,  cmap=plt.cm.gnuplot)
            plt.plot(x[el_pos], y[el_pos], 'bo')
            for i, e in enumerate(el_pos):
                plt.text(x[e], y[e], str(i+1), size=12)
            title_plot = 'Position:'+str(file_num)
            plt.title(title_plot)
            plt.axis('equal')
            plt.colorbar(im)
            filename = str(file_num)+'.png'
            plt.savefig(filename)
            #plt.show()
            plt.close()
            filenames.append(filename)
    else:
        for file_num in Y:
            ## Am Besten ist m_m zu Zero Reference
            toLoad = path+'/'+str(file_num)+'m_m.npy'
            IMGs = np.load(toLoad)
            for eb in range(IMGs.shape[0]):
                vis = IMGs[eb,:] ###!!!!!!!!!!!!!!!!!!!!!!!!attention
                g.update_reference(GroundTruth)
                #g.update_reference(np.zeros(192))
                baseline = g.eit_reconstruction(vis)
                if diff_img:
                    difference_image = g.eit_reconstruction(GroundTruth)
                else:
                    difference_image = g.eit_reconstruction(np.zeros(192))

            #difference_image = g.eit_reconstruction(vis)
                mesh_obj = g.mesh_obj;el_pos = g.el_pos;ex_mat = g.ex_mat
                pts = g.mesh_obj['node'];tri = g.mesh_obj['element']
                x   = pts[:, 0];y = pts[:, 1]
                #Print min and max impedance
                a = np.argmin(difference_image)
                b = np.argmax(difference_image)
                #print('Minimaler Wert: ',difference_image[a])
                #print('Maximaler Wert: ',difference_image[b])
                #print('Δ Permittivität: ',np.abs(difference_image[a]-difference_image[b]))
                shading = 'gouraud'
                shading = 'flat'
                #SHOW # 
                #fig, ax = plt.subplots(figsize=(10, 8))
                plt.figure(figsize=(10,8))
                im = plt.tripcolor(x , y , tri , difference_image, shading=shading, cmap=plt.cm.gnuplot)
                plt.tripcolor(x , y , tri , difference_image, shading=shading,  cmap=plt.cm.gnuplot)
                plt.plot(x[el_pos], y[el_pos], 'bo')
                for i, e in enumerate(el_pos):
                    plt.text(x[e], y[e], str(i+1), size=12)
                title_plot = 'Position:'+str(file_num)
                plt.title(title_plot)
                plt.axis('equal')
                plt.colorbar(im)
                filename = str(file_num)+'_'+str(eb)+'.png'
                plt.savefig(filename)
                #plt.show()
                plt.close()
                filenames.append(filename)
    # Build GIF
    NameGif = NameGif + '.gif'
    with imageio.get_writer(NameGif, mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    for ele in filenames:
        os.remove(ele)
        # save frame
        ### Anzeigen des Gifs
        # pick an animated gif file you have in the working directory
        
def show_gif(NameGif):
    NameGif = NameGif+'.gif'
    animation = pyglet.resource.animation(NameGif)
    sprite = pyglet.sprite.Sprite(animation)
    # create a window and set it to the image size
    win = pyglet.window.Window(width=sprite.width, height=sprite.height)
    # set window background color = r, g, b, alpha
    # each value goes from 0.0 to 1.0
    green = 0, 1, 0, 1
    pyglet.gl.glClearColor(*green)
    @win.event
    def on_draw():
        win.clear()
        sprite.draw()
    pyglet.app.run()