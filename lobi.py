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
import pyeit.eit.greit as greit
from pyeit.eit.fem import Forward
import pyeit.mesh as mesh
import reconstruction #Nicht über pip installierbar. Ist ein Package aus OpenEIT
import imageio

# Schriftformatierung der Plots
fs = 25
font = {'family': 'DejaVu Sans',
        'color':  'black',
        'weight': 'normal',
        'size': fs,}
plt.rcParams.update({'font.size': fs})
"""
Um dieses Modul nach einer Veränderung zu verwenden, muss der Kernel neu gestartet werden.
"""
###--------------------------------------------------------
def gen_env(mes,el,Schnkstp,leitw,temp,wasst,sonst='Keine weiteren Angaben'):
    """
    Initialisierung des Ordners mit Informationen zur Messung
    Input:  - mes      ... Dateiordnername
            - el       ... Elektrodenanzahl [16/32] -> Messmodus: ['d','e'] angegeben durch 16 oder 32 Elektroden
            - Schnkstp ... Schrittweite Schunk
            - leitw    ... Leitfähigkeit des Wassers
            - temp     ... Temperatur der Umgebung
            - wasst    ... Wasserstand im Messzylinder
            - sonst    ... Sonstige Informationen zur Messung
    Return:
            - Keine Rückgabe
    """
    print('Ordner mit dem Namen:"',mes,'" wurde erstellt.')
    if el == 16:
        mod = 'd'
    if el == 32:
        mod = 'e'
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
    Initialisieren des seriellen Ports. Port muss bekannt sein.
    Input:
        port ... Serieller Port des SpectraEIT-Kits, default = COM7
    Return:
        serialPort ... Fertige serielle Verbindung
    """
    serialPort = serial.Serial(port=port, baudrate=115200, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)
    print("Verbindung zu:" , port, "erfolgreich hergestellt.")
    return serialPort
###--------------------------------------------------------
def init_with_nel(port,Elek):
    """
    !!! Emfohlen wird die Funktion: init()
    Serielle Übermittlung und Initialisierung des Messmodusses für das SpectraEIT-Kit.
    Funktioniert leider nicht zuverlässig bei 32 Elektroden.
    """
    serialPort = serial.Serial(port=port, baudrate=115200, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)
    print("Verbindung zu:" , port, "erfolgreich hergestellt.")
    if Elek == 32:
        e = 'e'
        serialPort.write(e.encode())
        print("Messmodus: 'e'. Messung mit:",Elek,"Elektroden")
    else:
        d = 'd'
        serialPort.write(d.encode())
        print("Messmodus: 'e'. Messung mit:",Elek,"Elektroden")
    return serialPort

###--------------------------------------------------------
def parse_line(line):
    """
    Aus pyEIT zu verarbeitung der bitdaten.
    Für normalen Skriptgebrauch nicht benötigt.
    Input:
        line  ... Datensatz
    Return:
        items ... Elemente aus lines
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
def measure_data(N,serialPort,M=192,mode='d'):
    """
    Aufnahme von N Messwerten in einem definiterten Modus
    Input:  N    ... Anzahl der Messungen
            mode ... Modus des Spectra Kit
    Info:
            M+1 x N   ... Matrix, erste Spalte Messnummer.
            M         ... länge der Messung (default: 192)
            Matrix: 0 ... Messwerte
                    1 ... Messwerte
    Return:
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
def export_xlsx(A,path,mean,name='undefined'):
    """
    Exportieren der messdaten als Excel-Tebelle
    !Achtung: Bei der Weiterverarbeitung!
    Außerdem wird ein mittelwertbild abgezogen
    - raw ... Rohdaten ohne Abzug des Mittelwertes
    - m_m ... Rohdaten inklusive Abzug des Mittelwertes
    Input:
        A    ... Matrix aus Daten die exportiert werden soll
        path ... Verzeichnis in das exportiert werden soll
        mean ... Mittelwertbild der Messung
        name ... Name, unter dem die Messung gespeichert werden soll
    Return:
        - Keine Rückgabe
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
def CarCirc(R):
    """
    Umrechnung von kartesischen Koordinaten P(x,y) in
    Kreiskoordinatensystem r = sqrt(x^2+y^2). Dazu wird die
    Gewichtung in [%] für das Groundtruth Bild hinzugezogen.
    
    Input:  
        R ... P(x,y) mit R = [x1,y1,x2,y2,...xn,yn]. Es müssen immer die Tupel (x,y) gegeben sein!
    Return: 
        r ... Array mit prozentualer Positionierung
    """
    r = []
    for i in range(0,len(R),2):
        r.append(int(np.sqrt((R[i]**2+R[i+1]**2))*100))
    return np.array(r)
###--------------------------------------------------------
def ground_truth(objct,r,α,path,clockdirection=False,save_img=True):
    """
    Generiert das Groundtruth mit Koordinatensystem 2:
    https://github.com/JacobTh98/EIT_Measurement/blob/master/images/koordinate_system_2.png
    
    Input: Es müssen für object, r und α Arrays übergeben werden!
            objct ...'rectangle','circle','triangle'
            r    ... Radius 0...100 [%] vom Mittelpunkt
            α    ... Winkel 0...2π [Rad] von positiver x-Achse (default: gegen den Uhrzeigersinn)
            path ... Verzeichnis der Messng
            clockdirection ... im oder gegen Uhrzeigersinn drehen
           
           -> Für mehr Objekte Arrays für objct,r,α übergeben
    Return:
            IMG  ... Bild des Gegenstandes, positioniert im Einheitskreis.
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
        plt.rcParams.update({'font.size': fs})
        #im.save(path+"/"+str(objct)+str(r_old)+str(α_old)+".jpeg")
        im.save(path+"/"+"GroundTruth_α"+str(int(α))+".jpeg")
        np.save(path+"/"+"GroundTruth_np",IMG)
        print('Bild gespeichert')
    return IMG
###--------------------------------------------------------
def view_txt(Dir):
    """
    Ausgeben der info.txt durch Übergabe des Speicherverzeichnisses
        Input: 
            Dir ... Verzeichnis der Messung, indem "info.txt" abgelegt ist.
        Return:
            - Keine Rückgabe
    """
    txt = Dir+'/info.txt'
    file = open(txt)
    print(file.read())
###--------------------------------------------------------
def view_GrTr(Dir):
    """
    Ausgabe der GroundTruth
        Input: 
            Dir ... Verzeichnis der Messung, indem das Groundtruth-Bild abgelegt ist.
        Return:
            - Keine Rückgabe
    """
    Dir = Dir + '/GroundTruth_np.npy'
    img = np.load(Dir)
    plt.figure(figsize=(8,8))
    plt.title(r"Groundtruth", fontdict=font)
    plt.grid()
    plt.imshow(img)
###--------------------------------------------------------
def single_reconstruction(n_el,path,step,BackProj=True,diff_img=True,kind_of='m_m',save=False,DPI=300):
    """
    Zeigt das rekonstruierte Bild aus einem ausgewählten Datensatz.
    Für die Einzelrekonstruktion "single_img()" aufrufen.
    Hier ist auch nur GNM und SBP möglich. Für GREIT ebenfalls "single_img()" aufrufen.
    Input:
        n_el    ... Anzahl der Elektroden
        path    ... Verzeichnis der Messdaten
        step    ... Welche Iteration/Schritt wird ausgewertet
        BackProj... default: True, alternativ JacReconstruction
        diff_img... default: True = Es wird Ground Truth genommen, bei False zeros
        kind_of ... default: Meanfee: m_m. Alternative ist 'raw'
        save    ... default: False, Speichern des Bildes
        DPI     ... default: 300, Auflösung
    Return:
        - Keine Rückgabe
    """
    load_path = path+'/'+str(step)+kind_of+'.npy'
    IMGs = np.load(load_path)
    vis = IMGs[1,:]
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
    plt.rcParams.update({'font.size': fs})
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.tripcolor(x , y , tri , difference_image, shading=shading, cmap=plt.cm.viridis)
    ax.plot(x[el_pos], y[el_pos], 'bo')
    for i, e in enumerate(el_pos):
        ax.text(x[e], y[e], str(i+1), size=12)
    ax.axis('equal')
    plt.title(r"Rekonstruierte $\Delta$ Leitfähigkeit", fontdict=font)
    fig.colorbar(im)
    plt.show()
    if save:
        fig.savefig('single_rec'+'_'+str(n_el)+'.png', dpi=DPI)
###--------------------------------------------------------
def single_img(n_el,data,algorithm='GNM',save=False,DPI=300):
    """
    Zeigt das rekonstruierte Bild eines Datenvektors
    Input:
        n_el......... Anzahl der Elektroden
        data......... Datenvektor
        algorithm.... 'GNM', 'BP', 'GREIT'. default = 'GNM'
        save......... Speichern des Bilders. default = 'False'
        DPI.......... Auflösung. default = 300
    Return:
        - Keine Rückgabe
    Achtung: Wenn es nicht funktioniert "%matplotlib inline" in die Zeile über
             den Funktionsaufruf einfügen!
    """
    GroundTruth = np.ones(len(data))
    if algorithm == 'GREIT':
        g = reconstruction.GreitReconstruction(n_el)
        g.update_reference(GroundTruth)
        baseline = g.eit_reconstruction(GroundTruth)
        difference_image = g.eit_reconstruction(data)
        mesh_obj = g.mesh_obj;el_pos = g.el_pos;ex_mat = g.ex_mat
        pts = g.mesh_obj['node'];tri = g.mesh_obj['element']
        x   = pts[:, 0];y   = pts[:, 1]
        step=1
        mesh_new = mesh.set_perm(mesh_obj, background=1)
        fwd = Forward(mesh_obj, el_pos)
        f0 = fwd.solve_eit(ex_mat, step=step, perm=mesh_obj["perm"])
        f1 = fwd.solve_eit(ex_mat, step=step, perm=mesh_new["perm"])
        eit = greit.GREIT(mesh_obj, el_pos, ex_mat=ex_mat, step=step, parser="std")
        # parameter tuning is needed for better EIT images
        eit.setup(p=0.5, lamb=0.01)#
        ds = eit.solve(f1.v, f0.v, normalize=False)
        x, y, ds_greit = eit.mask_value(ds, mask_value=np.NAN)
        gr_max = np.max(np.abs(ds_greit))
        #%matplotlib inline
        plt.rcParams.update({'font.size': fs})
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(
            np.real(difference_image),
            interpolation="nearest",
            cmap=plt.cm.viridis#,
        )
        plt.title(r"Rekonstruierte $\Delta$ Leitfähigkeit", fontdict=font)
        ax.set_aspect("equal")
        fig.colorbar(im)
        plt.show()
    if algorithm == 'GNM' or algorithm == 'BP':
        if algorithm == 'GNM':
            g =reconstruction.JacReconstruction(n_el=n_el)
        else:
            g = reconstruction.BpReconstruction(n_el=n_el)
        g.update_reference(GroundTruth)
        baseline = g.eit_reconstruction(data)#####
        difference_image = g.eit_reconstruction(GroundTruth)
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
        #%matplotlib inline
        plt.rcParams.update({'font.size': fs})
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.tripcolor(x , y , tri , difference_image, shading=shading, cmap=plt.cm.viridis)
        ax.plot(x[el_pos], y[el_pos], 'bo')
        for i, e in enumerate(el_pos):
            ax.text(x[e], y[e], str(i+1), size=12)
        ax.axis('equal')
        fig.colorbar(im)
        plt.title(r"Rekonstruierte $\Delta$ Leitfähigkeit", fontdict=font)
        plt.show()
    if save:
        fig.savefig(str(algorithm)+'_'+str(n_el)+'.png', dpi=DPI)
###--------------------------------------------------------    
def gif_reconstruction(path,full=False,NameGif='Unnamed',BackProj=True,diff_img=True,kind_of='m_m'):
    """
    Input:
        path     ... Verzeichnis des Datensatzes
        full     ... default: False, True = Alle Daten rekonstruieren, False = nur aus jedem Messschritt ein Bild.
        NameGif  ... default: 'Unnamed', Name des gespeicherten Gifs
        BackProj ... default: True, alternativ JacReconstruction/GNM
        diff_img ... default: True = Es wird Ground Truth genommen, bei False Nullen
        kind_of  ... default: m_m = mittelwertfrei. Alternative: 'raw' = Mittelwertbehaftet
    Return:
        - Keine Rückgabe, Speichern des Gifs
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
    Y = np.arange(0,351,steps)
    #Y = np.arange(0,126,steps)
    filenames = []
    toLoad = path+'/Mean_empty_ground.npy'
    GroundTruth = np.load(toLoad)
    GroundTruth = np.zeros(192)
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
            plt.rcParams.update({'font.size': fs})
            plt.figure(figsize=(10,8))
            im = plt.tripcolor(x , y , tri , difference_image, shading=shading, cmap=plt.cm.viridis)
            plt.tripcolor(x , y , tri , difference_image, shading=shading,  cmap=plt.cm.viridis)
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
                #g.update_reference(GroundTruth)
                g.update_reference(np.zeros(192))
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
                plt.rcParams.update({'font.size': fs})
                #fig, ax = plt.subplots(figsize=(10, 8))
                plt.figure(figsize=(10,8))
                im = plt.tripcolor(x , y , tri , difference_image, shading=shading, cmap=plt.cm.viridis)
                plt.tripcolor(x , y , tri , difference_image, shading=shading,  cmap=plt.cm.viridis)
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
###--------------------------------------------------------        
def show_gif(NameGif):
    """
    Ausgeben des Gifs
    Input:
        NameGif ... Name des Gifs.
    Return:
        - Keine Rückgabe
    """
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
###--------------------------------------------------------