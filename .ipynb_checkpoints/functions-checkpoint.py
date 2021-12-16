import numpy as np
import pandas as pd
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
           α    ... Winkel von positiver x-Achse (gegen Uhrzeigersinn)
    Return:
           Bild des Gegenstandes, positioniert im Einheitskreis.
    """
    ###
    return r+α

###--------------------------------------------------------