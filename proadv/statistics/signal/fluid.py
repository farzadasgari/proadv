from proadv.statistics.spread import mean

def kinetic_turbulent_energy(u, v, w):

    up = u - mean(u)
    vp = v - mean(v)
    wp = w - mean(w)
    
    mean_ui2 = mean(up ** 2)
    mean_vi2 = mean(vp ** 2)
    mean_wi2 = mean(wp ** 2)

    k = 0.5 * (mean_ui2 + mean_vi2 + mean_wi2)

    return k
