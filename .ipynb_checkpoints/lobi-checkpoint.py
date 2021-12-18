import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import os
from datetime import datetime
"""
Um dieses Modul nach einer Veränderung zu verwenden, muss der Kernel neu gestartet werden.
"""
###--------------------------------------------------------
def gen_env(mes,el,mod):
    """
    Initialisierung des Ordners mit Informationen zur Messung
    Input:  - Elektrodenanzahl [16/32]
            - Messmodus: ['a','b','c','d','e']
    """
    print('Ordner mit dem Namen:"',mes,'" wurde erstellt.')
    os.makedirs(mes)
    dr = mes+'/info.txt'
    f = open(dr,"w")
    #Schreiben der Grunddaten
    f.write("EIT-Messung\n \n")
    d = "Datum: \t"+str(datetime.today().strftime('%Y-%m-%d %H:%M'))+"\n"
    m = "Messmodus: \t" + str(mod) + "\n"
    e = "Elektrodenanzahl: \t" +str(el)+"\n"
    f.write(d)
    f.write(m)
    f.write(e)
    f.close()
###--------------------------------------------------------
def mean_data(A):
    """
    Return: Vektor mit Mittelwerten der Zeilen, ohne die erste Spaltes
    """
    M = np.mean(A,0)
    M = M[1:] #Erste Spalte entfernen
    return M
###--------------------------------------------------------
def record_data(N, M = 192, mode = 'e'):
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
    MxN = np.c_[n,MxN]
    #Recording part...
      #...
    #Data handling
    return MxN

###--------------------------------------------------------
def export_xlsx(A,name ='undefined',transpose=True):
    """
    Exportieren der messdaten als Excel-Tebelle
    !Achtung: Bei der Weiterverarbeitung!
    """
    if transpose is True: # Invertieren der Messdaten
        df = pd.DataFrame(A).T
        df.to_excel(excel_writer = "test.xlsx")
    if transpose is False:
        df = pd.DataFrame(A)
        df.to_excel(excel_writer = "test.xlsx")
    
    print('Exported as:',name+str('.xlsx'))
###--------------------------------------------------------
def ground_truth(objct , r , α , dr ,clockdirection=False, save_img = True):
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
        im.save(dr+"/"+str(objct)+str(r_old)+str(α_old)+".jpeg")
        np.save(dr+"/"+"numpy",IMG)
        print('Bild gespeichert')
    return IMG
###--------------------------------------------------------