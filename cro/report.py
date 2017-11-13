import matplotlib.pyplot as plt
plt.style.use("ggplot")

def plot_results(Bestfitness, Meanfitness, title_info=None, filename=None):
    Ngen = len(Bestfitness) - 1
    ngen = range(Ngen+1)  
    fig, ax = plt.subplots(figsize=(10,10))
    ax.grid(True)
    ax.plot(ngen, Bestfitness, 'b')     
    ax.plot(ngen, Meanfitness, 'r--')           
    plt.xlabel('Number of generation')
    plt.xticks(ngen)
    plt.xlim(-1, Ngen+1)
    
    plt.legend(['Best fitness', 'Mean fitness'], loc="best")
    
    if title_info:
        L = title_info["L"]
        N = title_info["N"]
        M = title_info["M"]
        Fb = title_info["Fb"]
        Fa = title_info["Fa"]
        Fd = title_info["Fd"]
        Pd = title_info["Pd"]
    
        titlepro = ' Problem with Length vector (L): ' + str(L)
        titlepar = ", ".join(['Ngen: ' + str(Ngen),
                              'N: ' + str(N),
                              'M: ' + str(M),
                              'Fb: ' + str(Fb),
                              'Fa: ' + str(Fa),
                              'Fd: ' + str(Fd),
                              'Pd: ' + str(Pd)])

        plt.title(titlepro+'\n'+ titlepar)
    
    ax.scatter(Ngen, Bestfitness[-1], c='b')
    ax.annotate('Best: ' + str(Bestfitness[-1]) , (1.01*Ngen, 1.01*Bestfitness[-1]))
    
    if filename:
        fig.savefig(filename)
    else:
        plt.show()
