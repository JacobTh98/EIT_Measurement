import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import os
from datetime import datetime
import serial
from tqdm import tqdm
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