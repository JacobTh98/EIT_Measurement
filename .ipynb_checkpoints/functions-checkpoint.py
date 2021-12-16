import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
"""
Um dieses Modul nach einer Veränderung zu verwenden, muss der Kernel neu gestartet werden.
"""
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

def gen_ground_truth_pic(body,r,α):
    """
    Input: body ... 'rectangle','square', 'triangle'
           r    ... Radius vom Mittelpunkt
           α    ... Winkel von positiver x-Achse (im Uhrzeigersinn)
    Return:
           Bild des Gegenstandes, positioniert im Einheitskreis.
    """
    IMG=np.zeros((1000,1000))
    x_0 = y_0 =500 #Verschiebung Nullpunkt Koordinatensystem
    x = round(r*np.cos(α))
    y = round(r*np.sin(α))
    print('x:',x)
    print('y:',y)
    
    #              x,y      r     color.   strichdicke 
    cv.circle(IMG,(500,500),500,(255,255,0),1) #Rand
    if body == 'circle':
        cv.circle(IMG,(x_0+x,y_0+y),100,(255,0,0),-1)
    if body == 'rectangle':
        abw=100
        cv.rectangle(IMG,(x_0-abw+x,y_0-abw+y),(x_0+abw+x,y_0+abw+y),(255,0,0),-1)
    return IMG

###--------------------------------------------------------