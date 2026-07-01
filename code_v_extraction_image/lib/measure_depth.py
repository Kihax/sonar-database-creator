from typing import List
from matplotlib import pyplot as plt
import numpy as np
import pdb; 
import scipy.signal as signal
import pandas as pd
import skfuzzy as fuzz
from scipy import stats

def measure_depth(measure : List[float], sound_speed=1470):
    #plt.plot(np.flip(measure[0:s1]))

    #print("mu : ", mu)
    #print("sigma : ", sigma)

    p_pixel = 0;
    
    """for i in range(len(measure)):
        if(i >= len(measure) - 2):
            p_pixel  = i;
            break;
        m = measure[i]
        m1 = measure[i+1]
        m2 = measure[i+2]
        if(m > mu + 5*sigma and m1 > mu + 5*sigma and m2 > mu + 5*sigma):
            p_pixel = i;
            break;"""
    # 1. Gating temporel : on coupe le signal à la portée maximal

    z, p_pixel = mesure_seafloor_backtracking(measure, sound_speed)

    #if(z < 15):



    return z
    
    """fig, ax = plt.subplots()

    plt.plot(measure[s1:s2])

    plt.plot(measure[s1:s1+700]) # sur 10m de profondeur on suppose qu'on observe rien

    ax.axvline(x=i, color='red', linestyle='--', linewidth=2, label='Seuil (1680)')

    
    print("index profondeur : ",  i)
    print("profondeur : ", z)
    
    plt.show()"""

def premier_max_avec_delta(liste, delta=0.05):
    
    # 1. Trouver le maximum absolu
    max_absolu = max(liste)
    
    # 2. Définir le seuil
    seuil = max_absolu - delta
    
    # 3. Récupérer le premier élément (et son index) qui dépasse le seuil
    # next() renvoie le premier élément qui valide la condition
    try:
        index, valeur = next((i, x) for i, x in enumerate(liste) if x >= seuil)
        return index
    except StopIteration:
        return None

def create_fuzz_set(nb_classes, valeur_max):
    x = np.arange(0, valeur_max + 1, 1)
    centres = np.linspace(0, valeur_max, nb_classes)
    fuzzsets = []
    for i in range(nb_classes):
        if i == 0:
            debut, sommet, fin = centres[0], centres[0], centres[1]
        elif i == nb_classes - 1:
            debut, sommet, fin = centres[i-1], centres[i], centres[i]
        else:
            debut, sommet, fin = centres[i-1], centres[i], centres[i+1]
        fuzzsets.append(fuzz.trimf(x, [debut, sommet, fin]))
    return fuzzsets

def mesure_sol_fuzzyset(measure, sound_speed,dt = 1.152e-5):
    
    serie = pd.Series(measure)
    moyenne_glissante = serie.rolling(window=50, min_periods=1, center=True).mean().values

    valeur_max = int(max(moyenne_glissante))
    longueur_signal = len(moyenne_glissante)

    x_intensite = np.arange(0, valeur_max + 1, 1)
    x_time = np.arange(0, longueur_signal + 1, 1)

    # 1. Configuration des échelons (Facile à changer ici !)
    NB_ECHELLONS_INTENSITE = 10
    NB_ECHELLONS_TEMPS = 20

    intensite_fuzzsets = create_fuzz_set(NB_ECHELLONS_INTENSITE, valeur_max)
    time_fuzzsets = create_fuzz_set(NB_ECHELLONS_TEMPS, longueur_signal)

    # 2. CONFIGURATION DES RÈGLES (Modifiable à volonté)
    # Plus la valeur est proche de 1, plus cet échelon est autorisé à être le début du sol.
    # Ici : on bloque le tout début (0), on met le paquet sur les échelons 1 et 2 (Fort), 
    # puis on diminue (Moyen, Faible) au fur et à mesure que le temps passe.
    LOW_IMPACT = 0.05
    MEDIUM_IMPACT = 0.2
    HIGH_IMPACT=0.3
    poids_regle_temps = np.array([LOW_IMPACT, MEDIUM_IMPACT, HIGH_IMPACT, HIGH_IMPACT, HIGH_IMPACT, HIGH_IMPACT,  MEDIUM_IMPACT,  LOW_IMPACT, LOW_IMPACT, LOW_IMPACT, LOW_IMPACT, LOW_IMPACT, LOW_IMPACT, LOW_IMPACT,  LOW_IMPACT,  LOW_IMPACT,  LOW_IMPACT,  LOW_IMPACT,  LOW_IMPACT, LOW_IMPACT])
    
    # Sécurité si vous changez NB_ECHELLONS_TEMPS plus tard :
    if len(poids_regle_temps) != NB_ECHELLONS_TEMPS:
        # Recréer un profil par défaut (Rampe descendante après le début) si la taille ne colle pas
        poids_regle_temps = np.linspace(1.0, 0.0, NB_ECHELLONS_TEMPS)
        poids_regle_temps[0] = 0.0 # Toujours bloquer le premier échelon (bruit proche)

    # Profil d'intensité : on veut favoriser les échelons du haut (Fort, Très Fort)
    # [Très Faible, Faible, Moyen, Fort, Très Fort]
    poids_regle_intensite = np.array([0.2, 0.3, 0.5, 0.7, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8])

    scores_sol = np.zeros(longueur_signal)

    # 3. Évaluation du signal
    for j in range(longueur_signal):
        m = moyenne_glissante[j]
        
        # Calcul des appartenances pour le pixel courant
        deg_int = np.array([fuzz.interp_membership(x_intensite, intensite_fuzzsets[i], m) for i in range(NB_ECHELLONS_INTENSITE)])
        deg_time = np.array([fuzz.interp_membership(x_time, time_fuzzsets[i], j) for i in range(NB_ECHELLONS_TEMPS)])

        # Application des profils de règles par produit scalaire (pondération floue)
        # Cela calcule la corrélation entre "l'état du pixel" et "votre règle idéale"
        score_opportunite_temps = np.sum(deg_time * poids_regle_temps)
        score_force_intensite = np.sum(deg_int * poids_regle_intensite)

        # Combinaison : Il faut une bonne intensité ET que le temps soit opportun (ET flou = min)
        scores_sol[j] = min(score_force_intensite, score_opportunite_temps)

    # 4. DEFUBBIFICATION PAR GRADIENT (Pour choper le début du front montant)
    pente_scores = np.gradient(scores_sol)
    index_debut_sol = np.argmax(scores_sol)

    z = index_debut_sol*dt*sound_speed

    if(z > 0):
        # --- Graphique de contrôle ---
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        ax1.plot(measure, color='blue', label='Signal Sonar')
        ax1.axvline(x=index_debut_sol, color='red', linestyle='--', label='Début Sol Détecté')
        ax1.legend()
        ax1.set_title("Signal brut et détection")

        ax2.plot(moyenne_glissante, color='blue', label='Signal Sonar moyenne glissante')
        ax2.axvline(x=index_debut_sol, color='red', linestyle='--', label='Début Sol Détecté')
        ax2.legend()
        ax2.set_title("Signal brut et détection")

        ax3.plot(scores_sol, color='orange', label='Score flou du sol')
        ax3.plot(pente_scores, color='green', label='Gradient du score (Pente)')
        ax3.axvline(x=index_debut_sol, color='red', linestyle='--')
        ax3.legend()
        ax3.set_title("Analyse Floue & Défuzzification par Gradient")
        
        plt.tight_layout()
        plt.show()

    return z

def mesure_seafloor_backtracking(measure,  sound_speed,dt = 1.152e-5):
    serie = pd.Series(measure)
    measure_moving_average = serie.rolling(window=50, min_periods=1, center=True).mean().values

    serie_recherche_max = measure_moving_average[0:int(len(measure_moving_average)/2)]
    maximum = max(serie_recherche_max)
    p_pixel = np.argmax(serie_recherche_max)

    maximum_measure = max(measure[0:int(len(measure_moving_average)/2)])

    threshold = 0.2 * maximum;
    threhsold_measure = 0.01 * maximum_measure;

    #if(p_pixel == 0):
    #    return p_pixel
    
    while(measure_moving_average[p_pixel] > threshold or measure[p_pixel] > threhsold_measure):
        if(p_pixel == 0):
            return 0.0, 0
        p_pixel = p_pixel-1
        


    z = sound_speed * dt * p_pixel / 2

    
    if(z < 0):
        print("z : ", z)

        fig, (ax1, ax2, ax4) = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        ax1.plot(measure, color='blue', label='Signal Sonar')
        ax1.legend()
        ax1.axvline(x=p_pixel, color='red', linestyle='--', label='Début Sol Détecté')
        ax1.set_title("Signal brut et détection")

        ax2.plot(measure_moving_average, color='blue', label='Signal Sonar moyenne glissante')
        ax2.legend()
        ax2.axvline(x=p_pixel, color='red', linestyle='--', label='Début Sol Détecté')
        ax2.set_title("Signal brut et détection")
        

        ax4.plot(np.gradient(measure_moving_average), color='blue', label='Dérivé du signal')
        ax4.legend()
        ax4.axvline(x=p_pixel, color='red', linestyle='--', label='Début Sol Détecté')
        ax4.set_title("Signal brut et détection")

        plt.show()

    return z, p_pixel


def mesure_seafloor(measure,  sound_speed,dt = 1.152e-5):
    serie = pd.Series(measure)
    moyenne_glissante = serie.rolling(window=50, min_periods=1, center=True).mean().values
    integral = np.zeros(len(moyenne_glissante))
    
    for i in range(int(len(moyenne_glissante)/2)): # sur les exemples vuent, la réponse du sol ne dépassait pas la motié
        prec = integral[i-1]
        if(i == 0):
            prec = 0
        integral[i] = prec + moyenne_glissante[i]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    ax1.plot(measure, color='blue', label='Signal Sonar')
    ax1.legend()
    ax1.set_title("Signal brut et détection")

    ax2.plot(moyenne_glissante, color='blue', label='Signal Sonar moyenne glissante')
    ax2.legend()
    ax2.set_title("Signal brut et détection")

    ax3.plot(integral, color='blue', label='R**2')
    ax3.legend()
    ax3.set_title("Signal brut et détection")

    plt.show()

    return 0

def mesure_seafloor_linear_approximation(measure,  sound_speed,dt = 1.152e-5):
    serie = pd.Series(measure)
    moyenne_glissante = serie.rolling(window=50, min_periods=1, center=True).mean().values
    

    r2_list = np.zeros(len(measure))
    pentes = np.zeros(len(measure))
    for i in range(int(len(moyenne_glissante)/2)): # sur les exemples vuent, la réponse du sol ne dépassait pas la motié
        test = moyenne_glissante[i:i+100]
        x = np.arange(len(test))
        pente, ordonnee, r_value, p_value, std_err = stats.linregress(x, test)
        if(pente > 0):
            r2 = r_value**2
            r2_list[i] = r2
            pentes[i] = pente

    seuil = 0.75

    index_debut_sol = np.where(r2_list > seuil)[0]
    if(len(index_debut_sol) > 0):
        index_debut_sol = index_debut_sol[0]
    else:
        index_debut_sol = 0

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 6), sharex=True)
    ax1.plot(measure, color='blue', label='Signal Sonar')
    ax1.axvline(x=index_debut_sol, color='red', linestyle='--', label='Début Sol Détecté')
    ax1.legend()
    ax1.set_title("Signal brut et détection")

    ax2.plot(moyenne_glissante, color='blue', label='Signal Sonar moyenne glissante')
    ax2.axvline(x=index_debut_sol, color='red', linestyle='--', label='Début Sol Détecté')
    ax2.legend()
    ax2.set_title("Signal brut et détection")

    ax3.plot(r2_list, color='blue', label='R**2')
    ax3.axvline(x=index_debut_sol, color='red', linestyle='--', label='Début Sol Détecté')
    ax3.legend()
    ax3.set_title("Signal brut et détection")

    ax4.plot(pentes, color='blue', label='Pentes')
    ax4.legend()
    ax4.set_title("Signal brut et détection")

    plt.show()

    return 0


def mesure_seafloor_approxi_noise_normal_dist(measure):
    mu = measure[0:500].mean()
    sigma=  measure[0:500].std()
    for i in range(len(measure)):
        if(i >= len(measure) - 2):
            p_pixel  = i;
            break;
        m = measure[i]
        m1 = measure[i+1]
        m2 = measure[i+2]
        if(m > mu + 5*sigma and m1 > mu + 5*sigma and m2 > mu + 5*sigma):
            p_pixel = i;
            break;

def detect_bottom_fqi(signal_amplitude, threshold_snr=3.0):
    """
    Applique un FQI rapide pour trouver le premier retour du fond (Nadir).
    """

    amplitude = signal_amplitude
    
    # 2. Définition des fenêtres pour le FQI d'énergie (STA/LTA)
    len_sta = 30  # Fenêtre courte (taille du pulse)
    len_lta = 200 # Fenêtre longue (modèle du bruit de la colonne d'eau)
    
    # Calcul des énergies cumulées (méthode rapide par somme glissante)
    sta_energy = np.convolve(amplitude**2, np.ones(len_sta)/len_sta, mode='same')
    lta_energy = np.convolve(amplitude**2, np.ones(len_lta)/len_lta, mode='same')
    
    # 3. Calcul du FQI
    # On évite la division par zéro avec un petit epsilon
    fqi = sta_energy / (lta_energy + 1e-6)
    
    # 4. Recherche du moment de rupture (Seuil)
    # On cherche le premier index où le FQI dépasse le seuil
    detection_indices = np.where(fqi > threshold_snr)[0]
    
    if len(detection_indices) > 0:
        bottom_index = detection_indices[0] # Premier point de contact (Nadir)
        return bottom_index
    
    return None # Fond non détecté dans la portée maximale