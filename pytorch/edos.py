import matplotlib.pyplot as plt
import numpy as np


#presente = 0 
# direccion = f(y,t)

#presente = 0 
# direccion = f(y,t)
def get_euler(t0, tf, presente, direccion, dt, *args):  #t0, tf, presente, direccion, dt , *args
  values = []
  tiempos= []
  while t0 < tf:
    presente = presente + dt * direccion(t0,presente, *args)
    values.append(presente)
    tiempos.append(t0)
    t0 = t0 + dt
  values = np.array(values)
  tiempos = np.array(tiempos)
  return values,tiempos